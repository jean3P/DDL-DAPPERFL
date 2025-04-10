# src/models/fedprox/model.py

import copy
from ..utils.federated_model import FederatedModel
import torch.optim as optim
from tqdm import tqdm
import torch
import torch.nn as nn


class FedProx(FederatedModel):
    NAME = 'fedprox'

    def __init__(self, nets_list, args, transform):
        # Call the parent class constructor with the required arguments
        super(FedProx, self).__init__(nets_list, args, transform)
        self.args = args
        # Ensure that args has a device attribute
        if not hasattr(args, 'device'):
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = args.device
        self.mu = args.mu
        self.weight_decay = getattr(args, 'weight_decay', 0.0)
        self.nn = ANN(args=self.args, name='server').to(args.device)

    def ini(self):
        # Initialize all client networks with the global network weights.
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        if hasattr(self.args, "C"):
            m = max(int(self.args.C * self.args.parti_num), 1)
        else:
            m = self.args.parti_num
        online_clients = self.random_state.choice(total_clients, m, replace=False).tolist()
        self.online_clients = online_clients

        global_model = copy.deepcopy(self.global_net)
        for i in online_clients:
            self._train_net(i, self.nets_list[i], priloader_list[i], global_model)

        self.aggregate_nets(None)
        return None

    def _train_net(self, index, net, train_loader, global_model):
        net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss().to(self.device)
        iter_loader = tqdm(range(self.local_epoch), desc=f"Local Client {index}", leave=False)
        for _ in iter_loader:
            for batch_idx, (data, label) in enumerate(train_loader):
                data = data.to(self.device)
                label = label.to(self.device)
                outputs = net(data)
                loss = criterion(outputs, label)
                proximal_term = 0.0
                for w_local, w_global in zip(net.parameters(), global_model.parameters()):
                    proximal_term += ((w_local - w_global) ** 2).sum()
                loss += (self.mu / 2.0) * proximal_term
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_loader.set_description(f"Local Client {index} Loss: {loss.item():.3f}")


class ANN(nn.Module):
    def __init__(self, args, name):
        super(ANN, self).__init__()
        self.name = name
        self.len = 0
        self.loss = 0
        self.fc1 = nn.Linear(args.input_dim, 20)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, data):
        x = self.fc1(data)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)
        x = self.sigmoid(x)

        return x