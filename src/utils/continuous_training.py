# src/utils/continuous_training.py
# Fixed version with proper data partitioning

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
            domains_list (list): List of available domains
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

        # Track which subset of domain data each client uses
        self.client_domain_indices = {}
        self._initialize_domain_indices()

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

    def _initialize_domain_indices(self):
        """Initialize which subset index each client uses within their domain"""
        domain_client_counts = defaultdict(int)

        for client_id, domain in enumerate(self.client_domains):
            # Assign this client to the next available index for this domain
            self.client_domain_indices[client_id] = domain_client_counts[domain]
            domain_client_counts[domain] += 1

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

        # Count clients per domain after shifts
        new_domain_counts = defaultdict(int)
        for client_id, domain in enumerate(self.client_domains):
            if client_id not in clients_to_shift:
                new_domain_counts[domain] += 1

        for client_id in clients_to_shift:
            # Choose a new domain different from the current one
            current_domain = self.client_domains[client_id]
            available_domains = [d for d in self.domains_list if d != current_domain]
            new_domain = self.rng.choice(available_domains)

            # Update domain assignment
            self.client_domains[client_id] = new_domain

            # Assign subset index within new domain
            self.client_domain_indices[client_id] = new_domain_counts[new_domain]
            new_domain_counts[new_domain] += 1

            # Record this shift
            self.domain_shift_history[client_id].append(new_domain)

        return True

    def get_client_domain(self, client_id):
        """Get the current domain for a client"""
        return self.client_domains[client_id]

    def get_client_domain_and_index(self, client_id):
        """Get the current domain and subset index for a client"""
        return self.client_domains[client_id], self.client_domain_indices[client_id]

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


def prepare_domain_dataloaders_for_continuous_shift(private_dataset, domain_shifter, args):
    """
    Prepare proper dataloaders for each client based on their current domain assignment.
    This ensures each client gets a unique subset of their domain's data.

    This is the FIXED version that properly partitions data.
    """
    domains_list = private_dataset.DOMAINS_LIST
    current_train_loaders = []

    # Count how many clients are in each domain
    domain_client_mapping = defaultdict(list)
    for client_id in range(args.parti_num):
        domain = domain_shifter.get_client_domain(client_id)
        domain_client_mapping[domain].append(client_id)

    # For each domain, create properly partitioned dataloaders
    domain_loaders_cache = {}

    for domain, client_ids in domain_client_mapping.items():
        num_clients_in_domain = len(client_ids)

        if num_clients_in_domain > 0:
            # Create a list with this domain repeated for each client that needs it
            domain_list_for_partition = [domain] * num_clients_in_domain

            # Get properly partitioned loaders for this domain
            # This uses the dataset's built-in partitioning logic
            train_loaders, _ = private_dataset.get_data_loaders(domain_list_for_partition)

            # Map client IDs to their respective loaders
            for idx, client_id in enumerate(client_ids):
                domain_loaders_cache[client_id] = train_loaders[idx]

    # Build the final list of loaders in the correct order
    for client_id in range(args.parti_num):
        current_train_loaders.append(domain_loaders_cache[client_id])

    return current_train_loaders


def continuous_domain_shift_training(model, private_dataset, args):
    """
    Modified training function that supports continuous domain shifts.

    This is the FIXED version that uses proper data partitioning.
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

    # Get test loaders (these remain constant throughout training)
    _, test_loaders = private_dataset.get_data_loaders([])

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

        # Prepare dataloaders based on current domain assignments
        # This is the key fix - using proper partitioning
        current_train_loaders = prepare_domain_dataloaders_for_continuous_shift(
            private_dataset, domain_shifter, args
        )

        # Set the current train loaders for the model
        model.trainloaders = current_train_loaders

        # Perform local updates
        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(current_train_loaders)

        # Global evaluation on all test domains
        accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)

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


def global_evaluate(model, test_loaders, setting, name):
    """
    Evaluate the global model on all test loaders.
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
