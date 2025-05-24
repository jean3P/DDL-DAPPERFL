# src/models/mwfair.py

import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..utils.federated_model import FederatedModel


class MWFair(FederatedModel):
    NAME = 'mwfair'

    def __init__(self, nets_list, args, transform):
        super(MWFair, self).__init__(nets_list, args, transform)
        self.group_fairness = True
        self.fairness_lr = args.fairness_lr  # η_μ in the algorithm
        self.num_groups = args.num_groups

        # Initialize λ_gj for each group (Step 2 in Algorithm 1)
        self.lambda_groups = {g: 1.0 / self.num_groups for g in range(self.num_groups)}

        # Group probabilities P(G=g_j)
        self.group_probs = {g: 1.0 / self.num_groups for g in range(self.num_groups)}

        # Storage for group empirical risks (ε_gj)
        self.group_risks = {g: 0.0 for g in range(self.num_groups)}

        # For tracking group assignments
        self.client_groups = {}
        self.weight_decay = getattr(args, 'weight_decay', 1e-5)

    def ini(self):
        """Initialize the global model and distribute to clients"""
        self.global_net = copy.deepcopy(self.nets_list[0])
        global_w = self.nets_list[0].state_dict()
        for net in self.nets_list:
            net.load_state_dict(global_w)

        # Initialize client groups based on noise
        self._initialize_client_groups()

    def _initialize_client_groups(self):
        """Assign clients to groups based on noise (matching paper's setup)"""
        # Group 0: pristine, Group 1: noisy
        for i in range(self.args.parti_num):
            if i in self.args.noise_clients:
                self.client_groups[i] = 1  # Noisy group
            else:
                self.client_groups[i] = 0  # Pristine group

        # Update group probabilities based on actual distribution
        group_counts = {0: 0, 1: 0}
        for client, group in self.client_groups.items():
            group_counts[group] += 1

        for g in range(self.num_groups):
            self.group_probs[g] = group_counts[g] / self.args.parti_num

    def compute_group_empirical_risk(self, net, dataloader, client_idx):
        """Compute ε_gj(h) for each group - Step in Algorithm 1"""
        net.eval()
        group_losses = {g: 0.0 for g in range(self.num_groups)}
        group_counts = {g: 0 for g in range(self.num_groups)}

        # Get this client's group
        client_group = self.client_groups[client_idx]

        with torch.no_grad():
            total_loss = 0.0
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)

                # Add noise if this client has noise
                n_var = self.noise_variances.get(client_idx, 0.0)
                if n_var > 0.0:
                    sigma = n_var ** 0.5
                    data = data + torch.randn_like(data) * sigma

                output = net(data)
                loss = F.cross_entropy(output, target, reduction='mean')
                total_loss += loss.item()

        # Assign loss to client's group
        group_losses[client_group] = total_loss
        group_counts[client_group] = 1

        return group_losses

    def compute_group_weights(self):
        """Compute w_gj = λ_gj / P(G=g_j) - Step 5 in Algorithm 1"""
        group_weights = {}
        for g in range(self.num_groups):
            if self.group_probs[g] > 0:
                group_weights[g] = self.lambda_groups[g] / self.group_probs[g]
            else:
                group_weights[g] = 0.0
        return group_weights

    def update_lambda(self, group_risks):
        """Update λ_gj - Step 7 in Algorithm 1"""
        for g in range(self.num_groups):
            # λ_gj ← λ_gj · exp(-η_μ · ε_gj(h_ck))
            if group_risks[g] > 0:  # Only update if we have a risk value
                self.lambda_groups[g] *= np.exp(-self.fairness_lr * group_risks[g])

        # Normalize λ values to sum to 1
        total = sum(self.lambda_groups.values())
        if total > 0:
            for g in range(self.num_groups):
                self.lambda_groups[g] /= total

    def train_client_mw(self, client_idx, net, train_loader):
        """Train client using MW algorithm - Steps 5-6 in Algorithm 1"""
        # Step 5: Compute group weights
        group_weights = self.compute_group_weights()

        # Get client's group
        client_group = self.client_groups[client_idx]

        # Step 6: Find h_ck that minimizes weighted empirical risk
        net.to(self.device)
        net.train()
        optimizer = optim.SGD(net.parameters(), lr=self.local_lr, momentum=0.9, weight_decay=self.weight_decay)

        for epoch in range(self.local_epoch):
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Add noise if applicable
                n_var = self.noise_variances.get(client_idx, 0.0)
                if n_var > 0.0:
                    sigma = n_var ** 0.5
                    images = images + torch.randn_like(images) * sigma

                # Forward pass
                outputs = net(images)

                # Compute loss
                loss = F.cross_entropy(outputs, labels)

                # Weight the loss by group weight
                weighted_loss = loss * group_weights[client_group]

                # Backward and optimize
                optimizer.zero_grad()
                weighted_loss.backward()
                optimizer.step()

        # After training, compute group empirical risk
        group_risks = self.compute_group_empirical_risk(net, train_loader, client_idx)

        return group_risks

    def loc_update(self, priloader_list):
        """Main training loop following Algorithm 1"""
        total_clients = list(range(self.args.parti_num))
        online_clients = self.random_state.choice(total_clients, self.online_num, replace=False).tolist()
        self.online_clients = online_clients

        # Collect all group risks for lambda update
        all_group_risks = {g: [] for g in range(self.num_groups)}

        # For each client (Step 4 in Algorithm 1)
        for client_idx in online_clients:
            # Train and get group risks
            group_risks = self.train_client_mw(
                client_idx,
                self.nets_list[client_idx],
                priloader_list[client_idx]
            )

            # Collect risks by group
            for g in range(self.num_groups):
                if group_risks[g] > 0:
                    all_group_risks[g].append(group_risks[g])

        # Average risks across clients in each group
        avg_group_risks = {}
        for g in range(self.num_groups):
            if all_group_risks[g]:
                avg_group_risks[g] = np.mean(all_group_risks[g])
            else:
                avg_group_risks[g] = 0.0

        # Step 7: Update lambda values
        self.update_lambda(avg_group_risks)

        # Step 9: Server aggregation (FedAvg)
        self.aggregate_nets('weight')

        # Log current state
        print(f"Lambda values: {self.lambda_groups}")
        print(f"Group risks: {avg_group_risks}")

        return None
