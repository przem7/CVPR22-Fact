import argparse

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from .hypernetwork import HyperNetwork
from .util import get_average_embeddings
from utils import count_acc
from cw_torch.metric import cw_normality
from cw_torch.gamma import silverman_rule_of_thumb_sample

class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        if self.args.dataset in ['cifar100','manyshotcifar']:
            self.encoder = resnet20()
            self.num_features = 64
        if self.args.dataset in ['mini_imagenet','manyshotmini','imagenet100','imagenet1000']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.hypernet = HyperNetwork(args, self.num_features * self.args.way, self.num_features * self.args.way)

        self.pre_allocate = self.args.num_classes
        self.fc = nn.Linear(self.num_features, self.pre_allocate, bias=False)
        
        nn.init.orthogonal_(self.fc.weight)
        
    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, x, encoded=False):
        feature = self.encode(x)
        x = F.linear(F.normalize(feature, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))    
        x = self.args.temperature * x
        if not encoded:
            return x
        else:
            return x, feature

    def forward_delta(self, x, low, high, delta=None, encoded=False):
        feature = self.encode(x)
        if delta is None:
            weight = self.fc.weight
        else:
            weight = self.fc.weight
            delta_weight = torch.zeros_like(self.fc.weight)
            delta_weight[low:high] = delta
            weight = self.fc.weight + delta_weight
            
        x = F.linear(F.normalize(feature, p=2, dim=-1), F.normalize(weight, p=2, dim=-1))    
        x = self.args.temperature * x
        if not encoded:
            return x
        else:
            return x, feature

    def adapt(self, support_loader, query_loader, class_list, session):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.args.lr_new, momentum=0.9, dampening=0.9, weight_decay=0)

        test_class_high = self.args.base_class + session * self.args.way
        test_class_low = test_class_high - self.args.way

        with torch.enable_grad():
            for support_batch, query_batch in zip(support_loader, query_loader):
                support_data, support_label = [_.cuda() for _ in support_batch]
                query_data, query_label = [_.cuda() for _ in query_batch]
                
                support_feature = self.encode(support_data)
                
                support_avg_embeddings = self.get_average_embeddings(support_feature, support_label)
                support_avg_embeddings  = support_avg_embeddings.view(-1, self.num_features * self.args.way)

                delta = self.hypernet(support_avg_embeddings).view(self.args.way, self.num_features)

                logits = self.forward_delta(query_data, test_class_low, test_class_high, delta, False) # (25, 100)

                ce_loss = self.criterion(test_class_low, test_class_high, logits, query_label)
                total_loss = ce_loss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

                acc = count_acc(logits, query_label)
            
    def get_logits(self, x, fc):
        return F.linear(x, fc)

    def criterion(self, low_idx, high_idx, outputs, targets):
        return torch.nn.functional.cross_entropy(outputs[:, :high_idx], targets)

    def get_average_embeddings(self, embeddings, label):
        avg_enhanced_embeddings = []

        for l in set(label.cpu().numpy()):
            avg_enhanced_embeddings.append(
                embeddings[label==l].mean(dim=0)
            )

        avg_enhanced_embeddings = torch.stack(avg_enhanced_embeddings)
        
        return avg_enhanced_embeddings