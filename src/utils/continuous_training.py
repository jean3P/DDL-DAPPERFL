# src/utils/continuous_training.py

import copy
import numpy as np
import torch
from collections import defaultdict


class ContinuousDomainShift:
    """
    A class to manage continuous domain shifts for federated learning clients.
    This simulates real-world scenarios where a client's data distribution changes over time.
    """

    def __init__(self, domains_list, parti_num, shift_frequency=5, shift_ratio=0.3, seed=42):
        """
        Initialize the domain shift manager.

        Args:
            domains_list (list): List of available domains (e.g., ['caltech', 'amazon', 'webcam', 'dslr'])
            parti_num (int): Number of participants/clients
            shift_frequency (int): How often to shift domains (in communication rounds)
            shift_ratio (float): What proportion of clients will shift domains in each shift event
            seed (int): Random seed for reproducibility
        """
        self.domains_list = domains_list
        self.parti_num = parti_num
        self.shift_frequency = shift_frequency
        self.shift_ratio = shift_ratio
        self.rng = np.random.RandomState(seed)

        # Initialize client domain assignments
        self.client_domains = self._initial_assignment()

        # Track domain shifts for analysis
        self.domain_shift_history = defaultdict(list)
        for client_id in range(parti_num):
            self.domain_shift_history[client_id].append(self.client_domains[client_id])

    def _initial_assignment(self):
        """Create initial domain assignment for all clients"""
        # Ensure each domain is represented at least once
        initial_domains = []

        # Assign one client to each domain first
        for domain in self.domains_list:
            initial_domains.append(domain)

        # Randomly assign remaining clients
        remaining_clients = self.parti_num - len(self.domains_list)
        if remaining_clients > 0:
            random_domains = self.rng.choice(
                self.domains_list,
                size=remaining_clients,
                replace=True
            )
            initial_domains.extend(random_domains)

        # Shuffle the assignments
        self.rng.shuffle(initial_domains)

        return initial_domains

    def should_shift(self, round_idx):
        """Determine if domains should shift in the current round"""
        return round_idx > 0 and round_idx % self.shift_frequency == 0

    def update_domains(self, round_idx):
        """
        Update domain assignments based on the current round.
        Returns True if domains were shifted, False otherwise.
        """
        if not self.should_shift(round_idx):
            return False

        # Select clients to shift domains
        num_clients_to_shift = max(1, int(self.parti_num * self.shift_ratio))
        clients_to_shift = self.rng.choice(
            range(self.parti_num),
            size=num_clients_to_shift,
            replace=False
        )

        for client_id in clients_to_shift:
            # Choose a new domain different from the current one
            current_domain = self.client_domains[client_id]
            available_domains = [d for d in self.domains_list if d != current_domain]
            new_domain = self.rng.choice(available_domains)

            # Update domain assignment
            self.client_domains[client_id] = new_domain

            # Record this shift
            self.domain_shift_history[client_id].append(new_domain)

        return True

    def get_client_domain(self, client_id):
        """Get the current domain for a client"""
        return self.client_domains[client_id]

    def get_domain_distribution(self):
        """Return the current distribution of domains across clients"""
        distribution = {domain: 0 for domain in self.domains_list}
        for domain in self.client_domains:
            distribution[domain] += 1
        return distribution

    def print_shift_summary(self, round_idx):
        """Print a summary of domain shifts that occurred"""
        if self.should_shift(round_idx):
            print(f"\nDomain shift at round {round_idx}:")
            distribution = self.get_domain_distribution()
            for domain, count in distribution.items():
                print(f"  {domain}: {count} clients")


def continuous_domain_shift_training(model, private_dataset, args):
    """
    Modified training function that supports continuous domain shifts.

    Args:
        model: The federated learning model (e.g., DapperFL)
        private_dataset: The dataset class
        args: Command line arguments
    """
    # Set up the domain shift manager
    domains_list = private_dataset.DOMAINS_LIST
    domain_shifter = ContinuousDomainShift(
        domains_list=domains_list,
        parti_num=args.parti_num,
        shift_frequency=args.shift_frequency,
        shift_ratio=args.shift_ratio,
        seed=args.seed
    )

    # Initialize model and metrics tracking
    model.N_CLASS = private_dataset.N_CLASS
    accs_dict = {}
    mean_accs_list = []
    best_acc = 0
    best_accs = []

    # Get all data loaders for all domains
    domain_dataloaders = {}
    for domain in domains_list:
        # Get train and test loaders for this domain
        train_loaders, test_loaders = private_dataset.get_data_loaders([domain])
        domain_dataloaders[domain] = {
            'train': train_loaders,
            'test': test_loaders
        }

    # Initialize model
    if hasattr(model, 'ini'):
        model.ini()

    # Main training loop with domain shifts
    for epoch_index in range(args.communication_epoch):
        model.epoch_index = epoch_index

        # Check if domains should shift this round
        shift_occurred = domain_shifter.update_domains(epoch_index)
        if shift_occurred:
            domain_shifter.print_shift_summary(epoch_index)

        # Assign dataloader to each client based on their current domain
        current_train_loaders = []
        for client_id in range(args.parti_num):
            client_domain = domain_shifter.get_client_domain(client_id)
            # Get a dataloader from the client's current domain
            # Note: This is simplified - in practice, you'd need to ensure
            # different clients from the same domain get different data
            domain_loader = domain_dataloaders[client_domain]['train'][0]  # Using first loader for simplicity
            current_train_loaders.append(domain_loader)

        # Set the current train loaders for the model
        model.trainloaders = current_train_loaders

        # Perform local updates
        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(current_train_loaders)

        # Evaluate on all domains
        all_test_loaders = []
        for domain in domains_list:
            all_test_loaders.extend(domain_dataloaders[domain]['test'])

        # Global evaluation
        accs = global_evaluate(model, all_test_loaders)

        # Calculate and track metrics
        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)

        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]

        if mean_acc > best_acc:
            best_acc = mean_acc

        if len(best_accs) == 0:
            best_accs = copy.deepcopy(accs)

        for i in range(len(accs)):
            if accs[i] > best_accs[i]:
                best_accs[i] = accs[i]

        # Log metrics
        if args.wandb:
            import wandb
            wandb.log({
                "Best_Acc": best_acc,
                "Mean_Acc": mean_acc,
                "round": epoch_index
            })

            for i in range(len(accs)):
                name = "Domain" + str(i)
                wandb.log({
                    name + "_Acc": accs[i],
                    name + "_BestAcc": best_accs[i],
                    "round": epoch_index
                })

        # Print progress
        print(f'Round: {epoch_index}, Method: {model.args.model}, '
              f'Mean_Acc: {mean_acc}, Best_Acc: {best_acc}')
        print(f'Domain_Acc: {accs}, Domain_BestAcc: {best_accs}')

        # Track domain distribution for analysis
        if args.wandb:
            domain_dist = domain_shifter.get_domain_distribution()
            for domain, count in domain_dist.items():
                wandb.log({f"Clients_in_{domain}": count, "round": epoch_index})

    # Return final metrics
    return {
        'accs_dict': accs_dict,
        'mean_accs_list': mean_accs_list,
        'best_acc': best_acc,
        'domain_shift_history': domain_shifter.domain_shift_history
    }


def global_evaluate(model, test_loaders):
    """
    Evaluate the global model on all test loaders.
    Similar to the original global_evaluate function but simplified.
    """
    accs = []
    net = model.global_net
    status = net.training
    net.eval()

    for j, dl in enumerate(test_loaders):
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0

        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                if model.NAME == 'nefl':
                    outputs, _ = net(images)
                else:
                    outputs = net(images)

                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)

        top1acc = round(100 * top1 / total, 2)
        accs.append(top1acc)

    net.train(status)
    return accs
