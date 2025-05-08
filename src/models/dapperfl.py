# src/models/dapperfl.py

import torch.optim as optim
import torch.nn as nn
import torch.nn.utils.prune as torch_prune
from tqdm import tqdm
import copy
import numpy as np
from thop import profile
from torchstat import stat
from utils.args import *
import torch.nn.functional as F  
from .utils.federated_model import FederatedModel
import torch


class DapperFL(FederatedModel):
    NAME = 'dapperfl'

    def __init__(self, nets_list, args, transform):
        super(DapperFL, self).__init__(nets_list, args, transform)
        self.reg_coeff = args.reg_coeff
        self.alpha_0 = args.alpha
        self.alpha_min = args.alpha_min
        self.epsilon = args.epsilon
        self.pr_strategy = args.pr_strategy
        self.pr_ratios = ['0', '0.2', '0.4', '0.5', '0.6']

        # Group fairness components
        self.use_group_fairness = getattr(args, 'group_fairness', False)
        if self.use_group_fairness:
            self.fairness_lr = getattr(args, 'fairness_lr', 0.01)
            self.num_groups = getattr(args, 'num_groups', 2)
            # Initialize λ_gj for each group
            self.lambda_groups = {g: 1.0 / self.num_groups for g in range(self.num_groups)}
            # Group probabilities P(G=g_j)
            self.group_probs = {g: 1.0 / self.num_groups for g in range(self.num_groups)}
            # For tracking group assignments
            self.client_groups = {}  # Map from client index to group assignments

        self.prune_prob = {
            # Original model:
            '0': [0, 0, 0, 0],
            'AD': [0, 0, 0, 0],
            '0.1': [0.1, 0.1, 0.1, 0.1],
            '0.2': [0.2, 0.2, 0.2, 0.2],
            '0.3': [0.3, 0.3, 0.3, 0.3],
            '0.4': [0.4, 0.4, 0.4, 0.4],
            '0.5': [0.5, 0.5, 0.5, 0.5],
            '0.6': [0.6, 0.6, 0.6, 0.6],
            '0.7': [0.7, 0.7, 0.7, 0.7],
            '0.8': [0.8, 0.8, 0.8, 0.8],
            '0.9': [0.9, 0.9, 0.9, 0.9],
        }

    def ini(self):
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        # stat(self.global_net.cpu(), (3, 28, 28))
        for _, net in enumerate(self.nets_list):
            net.load_state_dict(global_w)

        # Initialize client groups if using group fairness
        if self.use_group_fairness and not hasattr(self.args, 'client_groups'):
            self._initialize_client_groups()

    def loc_update(self, priloader_list):
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        for i in online_clients:
            # If using group fairness, get group weights
            group_weights = self.compute_group_weights() if self.use_group_fairness else None

            # Train with group weights if using group fairness
            self._train_net(i, self.nets_list[i], priloader_list[i], group_weights)

            # If using group fairness, update lambda values
            if self.use_group_fairness:
                group_risks = self.compute_group_risks(self.nets_list[i], i, priloader_list[i])
                self.update_lambda(i, group_risks)

        # Aggregation
        self.aggregate_nets(None)

        return None

    def compute_group_weights(self):
        """Compute group importance weights w_gj = λ_gj / P(G=g_j)"""
        if not self.use_group_fairness:
            return None

        group_weights = {}
        for g in range(self.num_groups):
            group_weights[g] = self.lambda_groups[g] / self.group_probs[g]
        return group_weights

    def _initialize_client_groups(self):
        """Initialize client group assignments if not provided"""
        clients_per_group = self.args.parti_num // self.num_groups
        remainder = self.args.parti_num % self.num_groups

        group_assignments = []
        for g in range(self.num_groups):
            count = clients_per_group + (1 if g < remainder else 0)
            group_assignments.extend([g] * count)

        # Shuffle assignments
        np.random.shuffle(group_assignments)

        # Assign to clients
        self.client_groups = {i: group_assignments[i] for i in range(self.args.parti_num)}


    def _train_net(self, index, net, train_loader, group_weights=None):
        # net = net.cpu()
        # stat(net, (3, 32, 32))
        # print(list(net.named_buffers()))

        net = net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=1e-5)

        # Choose appropriate criterion based on whether we're using group fairness
        if self.use_group_fairness:
            criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
        else:
            criterion = nn.CrossEntropyLoss().to(self.device)

        iterator = tqdm(range(self.local_epoch))
        for i in iterator:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Add noise
                n_var = self.noise_variances.get(index, 0.0)
                if n_var > 0.0:
                    sigma = n_var ** 0.5
                    images = images + torch.randn_like(images) * sigma

                features = net.features(images)
                outputs = net.classifier(features)

                if self.use_group_fairness:
                    # Compute per-sample loss
                    losses = criterion(outputs, labels)

                    # Determine group for each sample (simplified)
                    # In real implementation, this would be based on data attributes
                    group_assignments = [batch_idx % self.num_groups] * len(labels)

                    # Apply group weights
                    weighted_losses = torch.zeros_like(losses)
                    for i, g in enumerate(group_assignments):
                        weighted_losses[i] = losses[i] * group_weights[g]

                    # Calculate final loss
                    loss = weighted_losses.mean()

                    # Add regularization if needed
                    if self.reg_coeff != 0.0:
                        reg = features.norm(dim=1).mean()
                        loss = loss + reg * self.reg_coeff
                else:
                    # Standard training without group fairness
                    if self.reg_coeff != 0.0:
                        loss = criterion(outputs, labels)
                        reg = features.norm(dim=1).mean()
                        loss = loss + reg * self.reg_coeff
                    else:
                        loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                iterator.desc = "Local Participant %d loss = %0.3f" % (index, loss)
                optimizer.step()

            if i == 0:
                # Co-Pruning
                if self.pr_strategy != "0":
                    # calculate co-weights
                    if self.alpha_0 != 0 and self.epoch_index != 0:
                        alpha_k = (1 - self.epsilon) ** self.epoch_index * self.alpha_0
                        if alpha_k < self.alpha_min:
                            alpha_k = self.alpha_min
                            # self.alpha_0 = 0
                        for [(name0, m0), (name1, m1)] in zip(self.global_net.named_modules(),
                                                              self.nets_list[index].named_modules()):
                            if isinstance(m1, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)) and torch_prune.is_pruned(m1):
                                m1.weight.data = alpha_k * m0.weight.data.clone() \
                                                 + (1 - alpha_k) * m1.weight.data.clone()
                    # pruning
                    if 'res' in self.nets_list[index].name:
                        self.nets_list[index] = self._res_pruning(index, self.nets_list[index])

    def _res_pruning(self, index, net, mod_struc=False):
        if self.pr_strategy == "progressive":
            dynamic_ratio = self._get_adaptive_progressive_ratio(index)
            pr_prob = [dynamic_ratio] * 4
        elif self.pr_strategy == "AD":
            pr_strategy = self.pr_ratios[index % len(self.pr_ratios)]  # get pruning ratio of specific client
            pr_prob = self.prune_prob[pr_strategy]  # get pruning ratios for layers
            self.prune_prob['AD'] = self.prune_prob[pr_strategy]  # copy pruning ratios for layers
        else:
            pr_prob = self.prune_prob[self.pr_strategy]  # get pruning ratios for layers

        if mod_struc:
            pr_prob = [0, 0, 0, 0]  # Do not prune the global model, just modify its structure for Pytorch processing.
        else:
            print("Prune local model %s, pr_ratio = %s" % (index, pr_prob))

        conv_count = 0
        down_count = 0  # r10's stage1 has no 'downsample' layer
        for name, module in net.named_modules():
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)) and torch_prune.is_pruned(module):
                torch_prune.remove(module, 'weight')
            if isinstance(module, nn.Conv2d):
                if conv_count == 0:  # The first conv layer in resnet.
                    conv_count += 1
                    continue
                if 'shortcut' in name:
                    # The first downsample conv layer, only prune 'out_planes'.
                    if down_count == 0:
                        # Use the pruning probability in stage1(pr_prob[0]) to prune 'out_planes'(dim=0).
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count], n=1, dim=0)
                        down_count += 1
                    else:  # The other downsample conv layer.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count - 1], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[down_count], n=1, dim=0)
                        down_count += 1
                    conv_count += 1
                    continue
                else:  # Normal conv layers in blocks.
                    # Stage1's 1st conv layer.
                    if conv_count == 1:
                        # Pruning 'out_planes'.
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=0)
                    # Stage1's other conv layers.
                    else:
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=1)
                        torch_prune.ln_structured(module, name="weight", amount=pr_prob[0], n=1, dim=0)
                    conv_count += 1
                    continue

            elif isinstance(module, nn.BatchNorm2d):
                # 'conv_count' in nn.BatchNorm2d is 1 bigger than nn.Conv2d.
                if conv_count == 1:  # The 1st bn in resnet.
                    continue
                torch_prune.l1_unstructured(module, name="weight", amount=pr_prob[2])

            elif isinstance(module, nn.Linear):
                torch_prune.ln_structured(module, name="weight", amount=pr_prob[-1], n=2, dim=1)

        # stat(glb_model, (3, 32, 32))
        # print(list(glb_model.named_buffers()))
        return net

    def _get_adaptive_progressive_ratio(self, client_index):
        """
        Compute adaptive progressive pruning ratio for a given client.
        Starts at the client's base ratio and decays toward 0 over time.
        """
        # Base pruning ratio for the client
        base_ratio = float(self.pr_ratios[client_index % len(self.pr_ratios)])

        # Decay factor over rounds (1 at round 0, decreasing over time)
        decay = (1 - self.epsilon) ** self.epoch_index
        return base_ratio * decay

    def compute_group_risks(self, net, client_idx, dataloader):
        """Compute empirical risk for each group on a client"""
        if not self.use_group_fairness:
            return None

        net.eval()
        group_risks = {g: 0.0 for g in range(self.num_groups)}
        group_counts = {g: 0 for g in range(self.num_groups)}

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                # For simplicity, we'll use a simple group assignment strategy
                # In a real implementation, this would be based on data attributes
                group = batch_idx % self.num_groups

                # Add noise if applicable
                n_var = self.noise_variances.get(client_idx, 0.0)
                if n_var > 0.0:
                    sigma = n_var ** 0.5
                    data = data + torch.randn_like(data) * sigma

                output = net(data)
                loss = F.cross_entropy(output, target, reduction='sum')

                group_risks[group] += loss.item()
                group_counts[group] += len(target)

        # Normalize risks by group sizes
        for g in range(self.num_groups):
            if group_counts[g] > 0:
                group_risks[g] /= group_counts[g]
            else:
                group_risks[g] = 0.0

        return group_risks

    def update_lambda(self, client_idx, group_risks):
        """Update λ_gj values using the MW update rule"""
        if not self.use_group_fairness:
            return

        for g in range(self.num_groups):
            # λ_gj ← λ_gj · exp(-η_μ · ε_gj(h_ck))
            self.lambda_groups[g] *= np.exp(-self.fairness_lr * group_risks[g])

        # Normalize λ values
        total = sum(self.lambda_groups.values())
        if total > 0:
            for g in range(self.num_groups):
                self.lambda_groups[g] /= total


