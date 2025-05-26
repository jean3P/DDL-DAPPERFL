# src/models/fedsr/model.py

import copy
import torch
from ..utils.federated_model import FederatedModel
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import torchvision.models as models


class FedSR(FederatedModel):
    NAME = 'fedsr'

    def __init__(self, nets_list, args, transform):
        # Call the parent class constructor with the required arguments
        super(FedSR, self).__init__(nets_list, args, transform)

        # Ensure that args has a device attribute
        if not hasattr(args, 'device'):
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.transform = transform
        self.device = args.device
        self.L2R_coeff = getattr(args, 'L2R_coeff', 0.01)

        self.nets_list = []
        for _ in range(args.parti_num):
            model = Base(args)
            self.nets_list.append(model)

        # Create a global network as a copy of the first client's network.
        self.global_net = copy.deepcopy(self.nets_list[0])

    def ini(self):
        # Initialize all client networks with the global network weights.
        global_w = self.global_net.state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        # Update all clients using their local loaders.
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients
        for i in range(self.args.parti_num):
            #self._train_net(i, self.nets_list[i], priloader_list[i], self.global_net)
            self.nets_list[i].train_client(priloader_list[i], steps=self.args.local_epoch)
        self.aggregate_nets(None)

    def _train_net(self, index, net, train_loader, global_model):
        net.to(self.device)
        net.train()
        data_iter = iter(train_loader)
        steps = self.args.local_epoch

        for step in range(steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)
            x, y = x.to(self.device), y.to(self.device)
            z = net.featurize(x)
            logits = net.cls(z)
            loss_cls = nn.functional.cross_entropy(logits, y)
            with torch.no_grad():
                global_features = global_model.featurize(x)
            loss_refine = nn.functional.mse_loss(z, global_features)
            loss = loss_cls + self.L2R_coeff * loss_refine
            net.optim.zero_grad()
            loss.backward()
            net.optim.step()
'''
    def aggregate_nets(self, freq=None):
        # A simple average aggregation across clients.
        global_w = self.global_net.state_dict()
        for key in global_w.keys():
            global_w[key] = 0
        for net in self.nets_list:
            net_w = net.state_dict()
            for key in global_w:
                global_w[key] += net_w[key] / self.args.parti_num
        self.global_net.load_state_dict(global_w)
        for net in self.nets_list:
            net.load_state_dict(global_w)
'''


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def average(self):
        return self.sum / self.count if self.count != 0 else 0


class Base(nn.Module):
    """
    FedSR Base model (deterministic version).
    This class is adapted from the FedSR implementation.
    """

    def __init__(self, args):
        super(Base, self).__init__()
        # Add the name attribute to match what's expected in main.py
        self.name = args.back_bone if hasattr(args, 'back_bone') else args.backbone

        # Ensure 'device' is defined in args; if not, assign a default.
        if not hasattr(args, 'device'):
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure 'z_dim' is defined; if not, set a default (e.g., 512)
        if not hasattr(args, 'z_dim'):
            args.z_dim = 512
        # Ensure 'back_bone' is defined; if not, copy from 'backbone'
        if not hasattr(args, 'back_bone'):
            args.back_bone = getattr(args, 'backbone', None)
            if args.back_bone is None:
                raise ValueError("Backbone is not specified in args.")
        # Ensure 'num_classes' is defined; if not, set a default (e.g., 10)
        if not hasattr(args, 'num_classes'):
            args.num_classes = 10
        # Ensure 'optim' is defined; if not, set a default (e.g., 'SGD')
        if not hasattr(args, 'optim'):
            args.optim = 'SGD'
        # Ensure 'lr' is defined; if not, set a default (e.g., 0.01)
        if not hasattr(args, 'lr'):
            args.lr = 0.01
        # Ensure 'weight_decay' is defined; if not, set a default (e.g., 1e-4)
        if not hasattr(args, 'weight_decay'):
            args.weight_decay = 1e-4
        # Optionally, ensure 'L2R_coeff' is defined; if not, set a default (e.g., 0.01)
        if not hasattr(args, 'L2R_coeff'):
            args.L2R_coeff = 0.01

        # Copy all parameters from args to self.
        for name in vars(args):
            setattr(self, name, getattr(args, name))
        out_dim = args.z_dim  # using deterministic representation

        if args.back_bone == 'resnet18':
            net = models.resnet18(pretrained=True)
            net.fc = nn.Linear(net.fc.in_features, out_dim)
        elif args.back_bone == 'resnet50':
            net = models.resnet50(pretrained=True)
            net.fc = nn.Linear(net.fc.in_features, out_dim)
        else:
            raise NotImplementedError("Backbone not implemented")

        self.net = net
        self.cls = nn.Linear(args.z_dim, args.num_classes)

        self.device = args.device
        self.net.to(self.device)
        self.cls.to(self.device)

        # Create an optimizer for all parameters
        if args.optim == 'SGD':
            self.optim = torch.optim.SGD(self.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        elif args.optim == 'Adam':
            self.optim = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError

        self.L2R_coeff = args.L2R_coeff

    def featurize(self, x, return_dist=False):
        # For deterministic FedSR, simply run the backbone
        return self.net(x)

    def forward(self, x):
        z = self.featurize(x)
        return self.cls(z)

    def train_client(self, loader, steps=1):
        self.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        reg_meter = AverageMeter()
        for _ in range(steps):
            x, y = next(iter(loader))
            x, y = x.to(self.device), y.to(self.device)
            z = self.featurize(x)
            logits = self.cls(z)
            loss = F.cross_entropy(logits, y)
            reg = z.norm(dim=1).mean()
            self.optim.zero_grad()
            (loss + reg * self.L2R_coeff).backward()
            self.optim.step()

            acc = (logits.argmax(1) == y).float().mean()
            loss_meter.update(loss.item(), x.shape[0])
            acc_meter.update(acc.item(), x.shape[0])
            reg_meter.update(reg.item(), x.shape[0])
        return {'acc': acc_meter.average(), 'loss': loss_meter.average(), 'reg': reg_meter.average()}

