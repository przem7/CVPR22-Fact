from .base import Trainer
import os.path as osp
import torch.nn as nn
from copy import deepcopy
import wandb
from .util import maybe_setup_wandb
import seaborn as sns
import matplotlib.pyplot as plt

from .helper import *
from utils import *
from dataloader.data_utils import *
from collections import OrderedDict


class PseudoParallelWrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return self.module(x, *args, **kwargs)


class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args, self.dataset = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)

        if self.args.device == "cuda":
            if self.args.num_gpu > 1:
                self.model = nn.DataParallel(self.model, None)
            else:
                self.model = PseudoParallelWrapper(self.model)
            self.model = self.model.cuda()
        else:
            self.model = PseudoParallelWrapper(self.model)
            self.model = self.model.cpu()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
            
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

        self.all_trainloaders = []
        maybe_setup_wandb(args)

    def get_optimizer_base(self):

        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step, gamma=self.args.gamma)
        elif self.args.schedule == 'Milestone':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args.milestones,
                                                             gamma=self.args.gamma)
        elif self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args, self.dataset)
            return trainset, trainloader, testloader
        else:
            trainset, trainloader, train_query_loader, test_support_loader, testloader = get_new_dataloader_for_hn(self.args, self.dataset, session)
            return trainset, trainloader, train_query_loader, test_support_loader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        # init train statistics
        result_list = [args]
        acc_matrix = []

        for session in range(args.start_session, args.sessions):
            self.model.load_state_dict(self.best_model_dict)

            if session == 0:  # load base class train img label
                train_set, trainloader, testloader = self.get_dataloader(session)
                self.all_trainloaders.append(trainloader)
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    # test model with all seen class
                    tsl, tsa = test(self.model, testloader, epoch, args, session)

                    # save better model
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    # populate metrics
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)

                    # logs
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))

                    # optimizer step
                    scheduler.step()

                # logs
                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                acc_matrix_row = self.test_tasks_separately()
                acc_matrix_row = acc_matrix_row + [0] * (args.sessions - len(acc_matrix_row))
                acc_matrix.append(acc_matrix_row)
                print(f'session {session} | acc matrix is:')
                print(acc_matrix)

            else:  # incremental learning sessions
                train_set, trainloader, train_query_loader, test_support_loader, testloader = self.get_dataloader(session)
                self.all_trainloaders.append((trainloader, train_query_loader))
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.train()
                
                trainloader.dataset.transform = testloader.dataset.transform
                for i in range(50):
                    self.model.module.adapt(trainloader, train_query_loader, np.unique(train_set.targets), session)

                self.model.eval()
                #tsl, tsa = test(self.model, testloader, 0, args, session,validation=False)
                #tsl, tsa = test_withfc(self.model, testloader, 0, args, session,validation=False)
                tsl, tsa, vcsa, sess_acc_list = self.test(test_support_loader, testloader,  session)

                acc_matrix_row = self.test_tasks_separately()
                acc_matrix_row = acc_matrix_row + [0] * (args.sessions - len(acc_matrix_row))
                acc_matrix.append(acc_matrix_row)
                print(f'session {session} | acc matrix is:')
                print(acc_matrix)
                
                # save model
                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                #torch.save(dict(params=self.model.state_dict()), save_model_dir)
                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        if os.environ.get("WANDB_ENTITY") is not None:
            self.log_acc_matrix(acc_matrix)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)


    def test(self, test_support_loader, testloader, session):
        # initialize indexes to separate currently considered session from prevoius one 
        high_idx = self.args.base_class + session * self.args.way
        low_idx = high_idx - self.args.way if (session > 0) else high_idx
        
        # evaluate model mode
        self.model.eval()
        
        # initialize averagers
        vl = Averager() # loss
        va = Averager() # total accuracy on all seen classes in the all sessions
        vcsa = Averager() # accuracy on current session's classes (current session accuracy)
        va5= Averager() # top 5 accuracy
        sess_averagers = OrderedDict()
        for i in range(session + 1):
            sess_averagers[f"session_{i}"] = Averager()

        with torch.no_grad():
            for i, batch in enumerate(testloader):
                data, test_label = [_.to(self.args.device) for _ in batch]
                    
                logits = self.model(data)
                logits_ = logits[:, :high_idx]

                loss = F.cross_entropy(logits_, test_label.long())
                acc = count_acc(logits_, test_label)

                if i < self.args.base_class:
                    sess_averagers["session_0"].add(acc)
                else:
                    N_SHOT = 5
                    sess_num = ((i - self.args.base_class) // N_SHOT) + 1
                    sess_averagers[f"session_{sess_num}"].add(acc)

                top5acc=count_acc_topk(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                va5.add(top5acc)

            vl = vl.item()
            va = va.item()
            va5= va5.item()
            vcsa = sess_averagers[f"session_{session}"].item()
            
            PARENT = 'few_shot_sessions'
            if os.environ.get("WANDB_ENTITY") is not None:
                wandb.log({
                        f'{PARENT}/test_loss': vl,
                        f'{PARENT}/test_accuracy': va,
                        f'{PARENT}/current_session_accuracy': vcsa
                      })
            print('sess {}, test, loss={:.4f} acc={:.4f}, acc@5={:.4f}, crr sess acc: {:.4f}'.format(session, vl, va, va5, vcsa))
        
        sess_acc_list = [ v.item() for k, v in sess_averagers.items() ]
        return vl, va, vcsa, sess_acc_list

    def test_tasks_separately(self):
        # evaluate model mode
        self.model.eval()

        # initialize averagers
        base_sess_averager = Averager()

        with torch.no_grad():
            for i, batch in enumerate(self.all_trainloaders[0]):
                data, label = [_.to(self.args.device) for _ in batch]

                logits = self.model(data)

                acc = count_acc(logits, label)

                base_sess_averager.add(acc)

        incremental_accuracies = []
        if len(self.all_trainloaders) > 1:
            for i, (loader_supp, loader_query) in enumerate(self.all_trainloaders[1:], 1):
                acc = self.model.module.test_adapt(loader_supp, loader_query, None, i)
                incremental_accuracies.append(acc)

        sess_acc_list = [base_sess_averager.item()] + incremental_accuracies
        return sess_acc_list

    def set_save_path(self):
        self.args.save_path = '%s/' % self.args.dataset
        self.args.save_path = self.args.save_path + '%s/' % self.args.project

        self.args.save_path = self.args.save_path + 'start_%d/' % (self.args.start_session)
        if self.args.schedule == 'Milestone':
            mile_stone = str(self.args.milestones).replace(" ", "").replace(',', '_')[1:-1]
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-MS_%s-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, mile_stone, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
            self.args.save_path = self.args.save_path + 'Bal%.2f-LossIter%d' % (
                self.args.balance, self.args.loss_iter)
        elif self.args.schedule == 'Step':
            self.args.save_path = self.args.save_path + 'Epo_%d-Lr_%.4f-Step_%d-Gam_%.2f-Bs_%d-Mom_%.2f' % (
                self.args.epochs_base, self.args.lr_base, self.args.step, self.args.gamma, self.args.batch_size_base,
                self.args.momentum)
        elif self.args.schedule == 'Cosine':
            self.args.save_path = self.args.save_path + 'Cosine-Epo_%d-Lr_%.4f' % (
                self.args.epochs_base, self.args.lr_base)
            self.args.save_path = self.args.save_path + 'Bal%.2f-LossIter%d' % (
                self.args.balance, self.args.loss_iter)

        if self.args.debug:
            self.args.save_path = os.path.join('debug', self.args.save_path)

        self.args.save_path = os.path.join('checkpoint', self.args.save_path)
        ensure_path(self.args.save_path)
        return None

    def log_acc_matrix(self, acc_matrix):
        acc_matrix = np.array(acc_matrix)

        fig, ax = plt.subplots(figsize=(11, 10))
        heatmap = sns.heatmap(acc_matrix, annot=True, ax=ax)

        image = wandb.Image(heatmap.get_figure(), caption="Accuracy per task heatmap")
          
        wandb.log({"Heatmaps": image})