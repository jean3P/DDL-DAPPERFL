# src/models/mwfair.py

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from ..utils.federated_model import FederatedModel


class MWFair(FederatedModel):
    NAME = 'mwfair'

    def __init__(self, nets_list, args, transform):
        super(MWFair, self).__init__(nets_list, args, transform)
        self.group_fairness = True
        self.fairness_lr = args.fairness_lr  # η_μ in the algorithm
        self.num_groups = args.num_groups

        # Initialize λ_gj for each group
        self.lambda_groups = {g: 1.0 / self.num_groups for g in range(self.num_groups)}

        # Group probabilities P(G=g_j)
        self.group_probs = {g: 1.0 / self.num_groups for g in range(self.num_groups)}

        # Storage for group empirical risks (ε_gj)
        self.group_risks = {g: 0.0 for g in range(self.num_groups)}

        # For tracking group assignments
        self.client_groups = {}  # Map from client index to group assignments
        self.weight_decay = getattr(args, 'weight_decay', 1e-5)

    def ini(self):
        """Initialize the global model and distribute to clients"""
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

        # Initialize client groups if not provided
        if not hasattr(self.args, 'client_groups'):
            # By default, assign clients to groups evenly
            self._initialize_client_groups()

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

    def compute_group_weights(self):
        """Compute group importance weights w_gj = λ_gj / P(G=g_j)"""
        group_weights = {}
        for g in range(self.num_groups):
            group_weights[g] = self.lambda_groups[g] / self.group_probs[g]
        return group_weights

    def update_lambda(self, client_idx, group_risks):
        """Update λ_gj values using the MW update rule"""
        for g in range(self.num_groups):
            # λ_gj ← λ_gj · exp(-η_μ · ε_gj(h_ck))
            self.lambda_groups[g] *= np.exp(-self.fairness_lr * group_risks[g])

        # Normalize λ values
        total = sum(self.lambda_groups.values())
        if total > 0:
            for g in range(self.num_groups):
                self.lambda_groups[g] /= total

    def compute_group_risks(self, net, client_idx, dataloader):
        """Compute empirical risk for each group on a client"""
        net.eval()
        group_risks = {g: 0.0 for g in range(self.num_groups)}
        group_counts = {g: 0 for g in range(self.num_groups)}

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(self.device), target.to(self.device)

                # Determine group for this batch (simplified approach)
                # In a real implementation, you'd need proper group assignments
                group = batch_idx % self.num_groups

                # Add noise if this client is in the noise_clients list
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

    def loc_update(self, priloader_list):
        """Update local models using the MW algorithm"""
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        # For each client in the online set
        for client_idx in online_clients:
            # Compute current group weights
            group_weights = self.compute_group_weights()

            # Train the local model with weighted loss
            self._train_net(client_idx, self.nets_list[client_idx], priloader_list[client_idx], group_weights)

            # Compute group risks after training
            group_risks = self.compute_group_risks(self.nets_list[client_idx], client_idx, priloader_list[client_idx])

            # Update lambda values using the MW update rule
            self.update_lambda(client_idx, group_risks)

        # Aggregate models using FedAvg
        self.aggregate_nets('weight')

        return None

    def _train_net(self, index, net, train_loader, group_weights):
        """Train the local model with weighted loss based on group importance weights"""
        net.to(self.device)
        net.train()

        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.weight_decay)
        criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)

        iter_loader = tqdm(range(self.local_epoch), desc=f"Local Client {index}", leave=False)
        for _ in iter_loader:
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Add noise if applicable
                n_var = self.noise_variances.get(index, 0.0)
                if n_var > 0.0:
                    sigma = n_var ** 0.5
                    images = images + torch.randn_like(images) * sigma

                # Forward pass
                outputs = net(images)

                # Compute per-sample loss
                losses = criterion(outputs, labels)

                # Determine group for each sample (simplified)
                # In real implementation, this would be based on data attributes
                group_assignments = [batch_idx % self.num_groups] * len(labels)

                # Apply group weights
                weighted_losses = torch.zeros_like(losses)
                for i, g in enumerate(group_assignments):
                    weighted_losses[i] = losses[i] * group_weights[g]

                # Take mean loss
                loss = weighted_losses.mean()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_loader.set_description(f"Local Client {index} Loss: {loss.item():.3f}")
