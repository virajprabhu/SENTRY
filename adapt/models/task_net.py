import sys
import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import torch.nn.functional as F
from torchvision import models

from .models import register_model
import numpy as np

sys.path.append('../../')
import utils

np.random.seed(1234)
torch.manual_seed(1234)

class TaskNet(nn.Module):

    num_channels = 3
    image_size = 32
    name = 'TaskNet'

    "Basic class which does classification."
    def __init__(self, num_cls=10, weights_init=None, l2_normalize=False, temperature=1.0):
        super(TaskNet, self).__init__()
        self.num_cls = num_cls
        self.l2_normalize = l2_normalize
        self.temperature = temperature
        self.setup_net()        
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, reverse_grad=False):
        # Extract features
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        x = x.clone()
        emb = self.fc_params(x)
        
        # Classify
        if isinstance(self.classifier, nn.Sequential): # LeNet   
            emb = self.classifier[:-1](emb)
            if reverse_grad: emb = utils.ReverseLayerF.apply(emb)
            if self.l2_normalize: emb = F.normalize(emb)
            score = self.classifier[-1](emb) / self.temperature
        else:                                          # ResNet
            if reverse_grad: emb = utils.ReverseLayerF.apply(emb)
            if self.l2_normalize: emb = F.normalize(emb)
            score = self.classifier(emb) / self.temperature
        
        return score

    def setup_net(self):
        """Method to be implemented in each class."""
        pass

    def load(self, init_path):
        net_init_dict = torch.load(init_path, map_location=torch.device('cpu'))
        self.load_state_dict(net_init_dict)

    def save(self, out_path):
        torch.save(self.state_dict(), out_path)

@register_model('LeNet')
class LeNet(TaskNet):
    "Network used for MNIST or USPS experiments."

    num_channels = 1
    image_size = 28
    l2_normalize = False
    name = 'LeNet'
    out_dim = 500 # dim of last feature layer

    def setup_net(self):

        self.conv_params = nn.Sequential(
                nn.Conv2d(self.num_channels, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )

        self.fc_params = nn.Linear(50*4*4, 500)
        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(500, self.num_cls, bias=True)
                )

@register_model('LeNetFS')
class LeNetFS(TaskNet):
    "Few shot LeNet"

    num_channels = 1
    l2_normalize = True
    image_size = 28
    name = 'LeNetFS'
    out_dim = 500 # dim of last feature layer

    def setup_net(self):

        self.conv_params = nn.Sequential(
                nn.Conv2d(self.num_channels, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                )

        self.fc_params = nn.Linear(50*4*4, 500)
        self.classifier = nn.Sequential(
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(500, self.num_cls, bias=False)
                )

@register_model('ResNet50')
class ResNet50(TaskNet):
    num_channels = 3
    name = 'ResNet50'

    def setup_net(self):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Identity()        
        self.classifier = nn.Linear(2048, self.num_cls)
        
        init.xavier_normal_(self.classifier.weight)
        self.classifier.bias.data.zero_()

@register_model('ResNet50FS')
class ResNet50FS(TaskNet):
    num_channels = 3
    name = 'ResNet50FS'

    def setup_net(self):
        model = models.resnet50(pretrained=True)
        model.fc = nn.Identity()
        self.conv_params = model
        self.fc_params = nn.Identity()        
        self.classifier = nn.Linear(2048, self.num_cls, bias=False)
        
        init.xavier_normal_(self.classifier.weight)
