import numpy as np
import torch
from dataloader.sampler import CategoriesSampler

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset =="manyshotcifar":
        import dataloader.cifar100.manyshot_cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    
    if args.dataset == 'manyshotcub':
        import dataloader.cub200.manyshot_cub as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = args.shot_num
        args.sessions = 11

    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'mini_imagenet_withpath':
        import dataloader.miniimagenet.miniimagenet_with_img as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    
    
    if args.dataset == 'manyshotmini':
        import dataloader.miniimagenet.manyshot_mini as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    
    if args.dataset == 'imagenet100':
        import dataloader.imagenet100.ImageNet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'imagenet1000':
        import dataloader.imagenet1000.ImageNet as Dataset
        args.base_class = 600
        args.num_classes=1000
        args.way = 50
        args.shot = 5
        args.sessions = 9

    return args, Dataset

def get_dataloader(args, session, dataset):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args, dataset)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, dataset, session)
    return trainset, trainloader, testloader

def get_base_dataloader(args, dataset):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':

        trainset = dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = dataset.ImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = dataset.ImageNet(root=args.dataroot, train=False, index=class_index)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
                                              num_workers=8, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)

    return trainset, trainloader, testloader



def get_base_dataloader_meta(args, dataset):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index)


    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=True)

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader(args, dataset, session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = dataset.ImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=True)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        testset = dataset.ImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, testloader

def get_new_dataloader_for_hn(args, dataset, session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    session_train_classes = get_session_train_classes(args, (session + 1))
    
    # support train data loader
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False, 
                                         train_tasks=args.train_tasks, shot=args.shot, few_shot=True,
                                         session_classes=session_train_classes)

    if args.dataset == 'cub200':
        trainset = dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = dataset.ImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)

    args.batch_size_new = args.way * args.shot

    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=False,
                                                num_workers=args.num_workers, pin_memory=True)


    # query train data loader
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        train_query_set = dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                           index=class_index, base_sess=False, 
                                           train_tasks=args.train_tasks, shot=args.shot, few_shot=True,
                                           session_classes=session_train_classes)

    train_query_loader = torch.utils.data.DataLoader(dataset=train_query_set, batch_size=args.batch_size_new, shuffle=False,
                                                     num_workers=args.num_workers, pin_memory=True)

    # support test data loader
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        test_support_set = dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                            index=class_index, base_sess=False, 
                                            train_tasks=args.train_tasks, shot=args.shot, few_shot=True,
                                            session_classes=session_train_classes)

    test_support_loader = torch.utils.data.DataLoader(dataset=test_support_set, batch_size=args.batch_size_new, shuffle=False,
                                                      num_workers=args.num_workers, pin_memory=True)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        testset = dataset.ImageNet(root=args.dataroot, train=False,
                                      index=class_new)

    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    return trainset, trainloader, train_query_loader, test_support_loader, testloader

def get_session_classes(args, session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list

def get_session_train_classes(args, session):
    class_list=np.arange(args.base_class + (session - 1) * args.way, args.base_class + session * args.way)
    return class_list