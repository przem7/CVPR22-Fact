import torch
import torch.nn as nn

class HyperNetwork(nn.Module):
    def __init__(self, args, num_features, out_features):
        super(HyperNetwork, self).__init__()

        self.args = args
        self.num_features = num_features
        self.out_features = out_features
        
        layers = []
        
        for i in range(self.args.hypernet_depth):
            in_features_number = self.num_features if (i == 0) else self.args.hypernet_width
            out_features_number = self.out_features if (i == self.args.hypernet_depth - 1) else self.args.hypernet_width
            
            layers.append(nn.Linear(in_features_number, out_features_number))
            
            if i < (self.args.hypernet_depth - 1):
                layers.append(nn.ReLU())

        # last activation layer
        layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)