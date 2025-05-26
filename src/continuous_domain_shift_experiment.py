# src/continuous_domain_shift_experiment.py

# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import datetime
import json

import torch
import numpy as np
import wandb
import argparse
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Import directly from our utilities
from utils.continuous_training import ContinuousDomainShift, prepare_domain_dataloaders_for_continuous_shift
from datasets import get_prive_dataset
from models import get_model
from utils.conf import set_random_seed
from utils.best_args import best_args
from utils.training import train
from utils.util import run_baseline_experiment

torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")
conf_path = os.getcwd()
sys.path.append('../../..')
sys.path.append(conf_path)
sys.path.append(conf_path + '/datasets')
sys.path.append(conf_path + '/backbone')
sys.path.append(conf_path + '/models')


def parse_args():
    parser = argparse.ArgumentParser(description='DapperFL with Continuous Domain Shift')

    # Model and dataset parameters
    parser.add_argument('--model', type=str, default='dapperfl', help='Name of FL framework')
    parser.add_argument('--dataset', type=str, default='fl_officecaltech', help='Dataset name')
    parser.add_argument('--parti_num', type=int, default=20, help='Number of participants')
    parser.add_argument('--communication_epoch', type=int, default=100, help='Total communication rounds')
    parser.add_argument('--local_epoch', type=int, default=5, help='Local training epochs')

    # DapperFL specific parameters
    parser.add_argument('--pr_strategy', type=str, default='AD', help='Model pruning strategy')
    parser.add_argument('--backbone', type=str, default='res18', help='Backbone architecture')
    parser.add_argument('--alpha', type=float, default=0.9, help='Co-pruning alpha coefficient')
    parser.add_argument('--alpha_min', type=float, default=0.1, help='Minimum alpha value')
    parser.add_argument('--epsilon', type=float, default=0.2, help='Epsilon coefficient')
    parser.add_argument('--reg_coeff', type=float, default=1e-2, help='L2 regularization coefficient')

    # Continuous domain shift parameters
    parser.add_argument('--shift_frequency', type=int, default=5,
                        help='How often domains shift (in communication rounds)')
    parser.add_argument('--shift_ratio', type=float, default=0.3,
                        help='Proportion of clients that shift domains in each shift event')
    parser.add_argument('--experiment_name', type=str, default='continuous_shift',
                        help='Name for this experiment')

    # General parameters
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--device_id', type=int, default=0, help='GPU device ID')
    parser.add_argument('--wandb', type=int, default=1, help='Enable wandb logging')
    parser.add_argument('--prefix', type=str, default='', help='Prefix for logs')
    parser.add_argument('--averaing', type=str, default='weight', help='Averaging strategy')
    parser.add_argument('--online_ratio', type=float, default=1.0, help='Ratio of online clients')
    parser.add_argument('--rand_dataset', type=int, default=1, help='Use random dataset')
    parser.add_argument('--csv_log', action='store_true', help='Enable CSV logging')

    # Noise-related parameters (needed by FederatedModel base class)
    parser.add_argument(
        '--noise_clients',
        nargs='+',
        type=int,
        default=[],
        help='Client indices with noise (empty by default for domain shift experiments)'
    )
    parser.add_argument('--noise_var', type=float, default=0.0, help='Noise variance')

    # Comparative experiment with baseline (no continuous shift)
    parser.add_argument('--run_baseline', action='store_true',
                        help='Run baseline experiment with no domain shifts')

    args = parser.parse_args()

    # Apply best hyperparameters from the best_args dictionary
    if args.dataset in best_args and args.model in best_args[args.dataset]:
        best = best_args[args.dataset][args.model]
        for key, value in best.items():
            setattr(args, key, value)

    # Ensure local_lr exists (needed by some models)
    if not hasattr(args, 'local_lr'):
        args.local_lr = 0.01

    return args


def setup_experiment(args, use_shift=True):
    """Setup experiment with or without continuous domain shift"""
    # Set random seed for reproducibility
    set_random_seed(args.seed)

    # Get dataset and model
    priv_dataset = get_prive_dataset(args)
    backbones_list = priv_dataset.get_backbone(args.parti_num, args.backbone)
    model = get_model(backbones_list, args, priv_dataset.get_transform())

    # Initialize wandb
    if args.wandb:
        prefix = args.prefix + '-' if args.prefix else ''
        shift_suffix = '-continuous' if use_shift else '-static'

        wandb.init(
            project="feddg-domain-shift",
            name=f"{prefix}{args.model}-{args.dataset}{shift_suffix}-{args.experiment_name}",
            config=args
        )

    return model, priv_dataset


def get_test_loaders_one_per_domain(priv_dataset):
    """Get exactly one test loader per domain for consistent evaluation"""
    domains_list = priv_dataset.DOMAINS_LIST
    test_loaders = []

    # Get test loaders for each domain separately
    for domain in domains_list:
        # Get loaders for this specific domain
        _, domain_test_loaders = priv_dataset.get_data_loaders([domain])
        # Take only the first test loader for this domain
        if domain_test_loaders and len(domain_test_loaders) > 0:
            test_loaders.append(domain_test_loaders[0])

    return test_loaders


def run_experiment_with_shift(model, priv_dataset, args, stats_file):
    """Run experiment with continuous domain shift and save statistics"""
    print("Running experiment with continuous domain shift")

    # Initialize comprehensive statistics tracking
    experiment_stats = {
        'config': vars(args),
        'domains_list': priv_dataset.DOMAINS_LIST,
        'rounds': [],
        'domain_shifts': [],
        'client_metrics': defaultdict(lambda: {
            'accuracies': [],
            'domain_history': [],
            'shift_rounds': []
        }),
        'summary': {}
    }

    # Setup statistics logging
    with open(stats_file, 'w') as f:
        # Write experiment configuration
        f.write("===== EXPERIMENT CONFIGURATION =====\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Backbone: {args.backbone}\n")
        f.write(f"Participants: {args.parti_num}\n")
        f.write(f"Communication Epochs: {args.communication_epoch}\n")
        f.write(f"Local Epochs: {args.local_epoch}\n")
        f.write(f"Shift Frequency: {args.shift_frequency}\n")
        f.write(f"Shift Ratio: {args.shift_ratio}\n")
        f.write(f"Pruning Strategy: {args.pr_strategy}\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Alpha Min: {args.alpha_min}\n")
        f.write(f"Epsilon: {args.epsilon}\n")
        f.write(f"Reg Coefficient: {args.reg_coeff}\n\n")

        f.write("===== DOMAIN SHIFT EXPERIMENT RESULTS =====\n")
        f.write("Format: Round [X] | Mean Acc: [Y]% | Domain Accs: [Z] | Shifted: [Yes/No]\n\n")

    # Set up the domain shift manager
    domains_list = priv_dataset.DOMAINS_LIST
    domain_shifter = ContinuousDomainShift(
        domains_list=domains_list,
        parti_num=args.parti_num,
        shift_frequency=args.shift_frequency,
        shift_ratio=args.shift_ratio,
        seed=args.seed
    )

    # Initialize model and metrics tracking
    model.N_CLASS = priv_dataset.N_CLASS
    accs_dict = {}
    mean_accs_list = []
    best_acc = 0
    best_accs = []

    # Track metrics before and after shifts
    pre_shift_metrics = []
    post_shift_metrics = []

    # Get test loaders - one per domain
    test_loaders = get_test_loaders_one_per_domain(priv_dataset)
    print(f"Number of test loaders: {len(test_loaders)}, Number of domains: {len(domains_list)}")

    # Initialize model
    if hasattr(model, 'ini'):
        model.ini()

    # Main training loop with domain shifts
    for epoch_index in range(args.communication_epoch):
        model.epoch_index = epoch_index
        round_stats = {
            'round': epoch_index,
            'domain_shift_occurred': False,
            'clients_shifted': [],
            'domain_distribution': {},
            'accuracies': {},
            'mean_accuracy': 0
        }

        # Check if domains should shift this round
        shift_occurred = domain_shifter.update_domains(epoch_index)
        round_stats['domain_shift_occurred'] = shift_occurred

        if shift_occurred:
            domain_shifter.print_shift_summary(epoch_index)

            # Track which clients shifted
            for client_id in range(args.parti_num):
                history = domain_shifter.domain_shift_history[client_id]
                if len(history) > 1 and history[-1] != history[-2]:
                    round_stats['clients_shifted'].append(client_id)
                    experiment_stats['client_metrics'][client_id]['shift_rounds'].append(epoch_index)

            # Record domain distribution
            round_stats['domain_distribution'] = domain_shifter.get_domain_distribution()

            # Save domain shift information to stats file
            with open(stats_file, 'a') as f:
                f.write(f"===== DOMAIN SHIFT AT ROUND {epoch_index} =====\n")
                f.write(f"Clients shifted: {round_stats['clients_shifted']}\n")
                f.write("New domain distribution:\n")
                for domain, count in round_stats['domain_distribution'].items():
                    f.write(f"  {domain}: {count} clients\n")
                f.write("\n")

        # Prepare dataloaders based on current domain assignments using the correct approach
        current_train_loaders = prepare_domain_dataloaders_for_continuous_shift(
            priv_dataset, domain_shifter, args
        )

        # Set the current train loaders for the model
        model.trainloaders = current_train_loaders

        # Update client metrics
        for client_id in range(args.parti_num):
            client_domain = domain_shifter.get_client_domain(client_id)
            experiment_stats['client_metrics'][client_id]['domain_history'].append(client_domain)

        # Perform local updates
        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(current_train_loaders)

        # Global evaluation
        accs = global_evaluate(model, test_loaders, priv_dataset.SETTING, priv_dataset.NAME)

        # Ensure we have the correct number of accuracies
        if len(accs) != len(domains_list):
            print(f"Warning: Expected {len(domains_list)} accuracies but got {len(accs)}")
            # Truncate or pad as necessary
            if len(accs) > len(domains_list):
                accs = accs[:len(domains_list)]
            else:
                # Pad with zeros if we have fewer accuracies than domains
                accs.extend([0.0] * (len(domains_list) - len(accs)))

        # Calculate and track metrics
        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)
        round_stats['mean_accuracy'] = mean_acc

        # Record per-domain accuracies
        domain_accs = {}
        for i in range(len(domains_list)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]
            domain_accs[i] = accs[i]
            round_stats['accuracies'][domains_list[i]] = accs[i]

        # Track metrics around shifts
        if shift_occurred and epoch_index > 0:
            pre_shift_metrics.append({
                'round': epoch_index,
                'pre_shift_acc': mean_accs_list[epoch_index - 1] if epoch_index > 0 else 0,
                'post_shift_acc': mean_acc
            })

        # Track best accuracy
        if mean_acc > best_acc:
            best_acc = mean_acc

        if len(best_accs) == 0:
            best_accs = accs.copy()
        else:
            for i in range(len(accs)):
                if accs[i] > best_accs[i]:
                    best_accs[i] = accs[i]

        # Save round statistics
        experiment_stats['rounds'].append(round_stats)

        # Write to stats file
        with open(stats_file, 'a') as f:
            f.write(
                f"Round {epoch_index} | Mean Acc: {mean_acc:.2f}% | Domain Accs: {accs} | Shifted: {'Yes' if shift_occurred else 'No'}\n")
            if mean_acc > best_acc - 0.001:  # Account for floating point precision
                f.write(f"  ★ New best accuracy: {mean_acc:.2f}%\n")

        # Log metrics
        if args.wandb:
            wandb.log({
                "Best_Acc": best_acc,
                "Mean_Acc": mean_acc,
                "round": epoch_index
            })

            for i in range(len(domains_list)):
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

    # Calculate comprehensive statistics
    experiment_stats['summary'] = calculate_comprehensive_stats(
        mean_accs_list,
        experiment_stats['rounds'],
        args.shift_frequency,
        domain_shifter.domain_shift_history
    )

    # Write final summary statistics
    with open(stats_file, 'a') as f:
        f.write("\n===== EXPERIMENT SUMMARY =====\n")
        f.write(f"Best Mean Accuracy: {best_acc:.2f}%\n")
        f.write(f"Best Domain Accuracies: {best_accs}\n")
        f.write(f"Average Accuracy: {experiment_stats['summary']['avg_accuracy']:.2f}%\n")
        f.write(f"Accuracy Std Dev: {experiment_stats['summary']['accuracy_std']:.4f}\n")
        f.write(f"Number of Domain Shifts: {experiment_stats['summary']['num_shifts']}\n")

        if experiment_stats['summary']['avg_drop_after_shift'] is not None:
            f.write(f"Average Accuracy Drop After Shift: {experiment_stats['summary']['avg_drop_after_shift']:.4f}\n")
            f.write(f"Average Recovery Rounds: {experiment_stats['summary']['avg_recovery_rounds']:.2f}\n")

        f.write(f"\nConvergence Metrics:\n")
        f.write(f"  Final 10 Rounds Avg: {experiment_stats['summary']['final_10_avg']:.2f}%\n")
        f.write(f"  Final 10 Rounds Std: {experiment_stats['summary']['final_10_std']:.4f}\n")

        f.write("\nDomain distribution at the end of experiment:\n")
        for domain, count in domain_shifter.get_domain_distribution().items():
            f.write(f"  {domain}: {count} clients\n")

    # Save detailed statistics as JSON
    json_stats_file = stats_file.replace('.txt', '_detailed.json')
    with open(json_stats_file, 'w') as f:
        json.dump(experiment_stats, f, indent=2, default=str)

    # Create visualizations
    create_comprehensive_visualizations(experiment_stats, args, stats_file)

    # Return final metrics
    return {
        'accs_dict': accs_dict,
        'mean_accs_list': mean_accs_list,
        'best_acc': best_acc,
        'domain_shift_history': domain_shifter.domain_shift_history,
        'experiment_stats': experiment_stats
    }


def calculate_comprehensive_stats(mean_accs_list, rounds, shift_frequency, domain_shift_history):
    """Calculate comprehensive statistics for the experiment"""
    stats = {
        'avg_accuracy': np.mean(mean_accs_list),
        'accuracy_std': np.std(mean_accs_list),
        'max_accuracy': np.max(mean_accs_list),
        'min_accuracy': np.min(mean_accs_list),
        'num_shifts': sum(1 for r in rounds if r['domain_shift_occurred']),
        'final_10_avg': np.mean(mean_accs_list[-10:]) if len(mean_accs_list) >= 10 else np.mean(mean_accs_list),
        'final_10_std': np.std(mean_accs_list[-10:]) if len(mean_accs_list) >= 10 else np.std(mean_accs_list),
    }

    # Calculate accuracy drops after shifts
    accuracy_drops = []
    recovery_rounds = []

    for i, round_info in enumerate(rounds):
        if round_info['domain_shift_occurred'] and i > 0:
            pre_shift_acc = mean_accs_list[i - 1]
            post_shift_acc = mean_accs_list[i]
            drop = pre_shift_acc - post_shift_acc
            accuracy_drops.append(drop)

            # Find recovery rounds (rounds to reach pre-shift accuracy again)
            recovery = 0
            for j in range(i + 1, len(mean_accs_list)):
                recovery += 1
                if mean_accs_list[j] >= pre_shift_acc:
                    break
            recovery_rounds.append(recovery)

    if accuracy_drops:
        stats['avg_drop_after_shift'] = np.mean(accuracy_drops)
        stats['max_drop_after_shift'] = np.max(accuracy_drops)
        stats['avg_recovery_rounds'] = np.mean(recovery_rounds)
    else:
        stats['avg_drop_after_shift'] = None
        stats['max_drop_after_shift'] = None
        stats['avg_recovery_rounds'] = None

    # Calculate client stability (how often clients change domains)
    client_stability = {}
    for client_id, history in domain_shift_history.items():
        changes = sum(1 for i in range(1, len(history)) if history[i] != history[i - 1])
        client_stability[client_id] = changes

    stats['avg_client_domain_changes'] = np.mean(list(client_stability.values()))
    stats['max_client_domain_changes'] = np.max(list(client_stability.values()))

    return stats


def create_comprehensive_visualizations(experiment_stats, args, stats_file):
    """Create comprehensive visualizations for the experiment"""
    viz_dir = "visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    base_name = f"{args.model}_{args.dataset}_{args.pr_strategy}_{args.shift_frequency}_{args.shift_ratio}"

    # 1. Accuracy over time with shift markers
    plt.figure(figsize=(14, 8))

    rounds = [r['round'] for r in experiment_stats['rounds']]
    mean_accs = [r['mean_accuracy'] for r in experiment_stats['rounds']]

    # Plot mean accuracy
    plt.plot(rounds, mean_accs, 'b-', linewidth=2, label='Mean Accuracy')

    # Mark domain shifts
    shift_rounds = [r['round'] for r in experiment_stats['rounds'] if r['domain_shift_occurred']]
    for sr in shift_rounds:
        plt.axvline(x=sr, color='red', linestyle='--', alpha=0.5)

    # Add a dummy line for legend
    plt.axvline(x=-1, color='red', linestyle='--', alpha=0.5, label='Domain Shift')

    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(
        f'Accuracy Evolution with Domain Shifts\n(Strategy: {args.pr_strategy}, Shift Freq: {args.shift_frequency}, Shift Ratio: {args.shift_ratio})',
        fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    accuracy_plot = f"{viz_dir}/{base_name}_accuracy.png"
    plt.savefig(accuracy_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Domain distribution heatmap
    plt.figure(figsize=(14, 8))

    # Create matrix for domain distribution over time
    domains = experiment_stats['domains_list']
    domain_matrix = np.zeros((len(domains), len(rounds)))

    for i, round_info in enumerate(experiment_stats['rounds']):
        if round_info['domain_distribution']:
            for j, domain in enumerate(domains):
                domain_matrix[j, i] = round_info['domain_distribution'].get(domain, 0)

    # Plot heatmap
    sns.heatmap(domain_matrix,
                xticklabels=[r if r % 10 == 0 else '' for r in rounds],
                yticklabels=domains,
                cmap='YlOrRd',
                cbar_kws={'label': 'Number of Clients'},
                annot=False)

    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Domain', fontsize=12)
    plt.title(
        f'Domain Distribution Over Time\n(Strategy: {args.pr_strategy}, Shift Freq: {args.shift_frequency}, Shift Ratio: {args.shift_ratio})',
        fontsize=14)
    plt.tight_layout()

    distribution_plot = f"{viz_dir}/{base_name}_distribution.png"
    plt.savefig(distribution_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Per-domain accuracy evolution
    plt.figure(figsize=(14, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(domains)))

    for i, domain in enumerate(domains):
        domain_accs = [r['accuracies'].get(domain, 0) for r in experiment_stats['rounds']]
        plt.plot(rounds, domain_accs, color=colors[i], linewidth=2, label=f'{domain}')

    # Mark domain shifts
    for sr in shift_rounds:
        plt.axvline(x=sr, color='gray', linestyle='--', alpha=0.3)

    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(
        f'Per-Domain Accuracy Evolution\n(Strategy: {args.pr_strategy}, Shift Freq: {args.shift_frequency}, Shift Ratio: {args.shift_ratio})',
        fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    domain_accuracy_plot = f"{viz_dir}/{base_name}_domain_accuracy.png"
    plt.savefig(domain_accuracy_plot, dpi=300, bbox_inches='tight')
    plt.close()

    # Update stats file with visualization paths
    with open(stats_file, 'a') as f:
        f.write(f"\n===== VISUALIZATIONS =====\n")
        f.write(f"Accuracy plot: {accuracy_plot}\n")
        f.write(f"Distribution plot: {distribution_plot}\n")
        f.write(f"Domain accuracy plot: {domain_accuracy_plot}\n")

    if args.wandb:
        wandb.log({
            "accuracy_plot": wandb.Image(accuracy_plot),
            "distribution_plot": wandb.Image(distribution_plot),
            "domain_accuracy_plot": wandb.Image(domain_accuracy_plot)
        })


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


def main():
    args = parse_args()

    # Create directory for stats files if it doesn't exist
    stats_dir = "experiment_stats"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    # Create a timestamp for the experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a stats file for the experiment
    stats_file = f"{stats_dir}/{args.model}_{args.dataset}_{args.pr_strategy}_shift{args.shift_frequency}_ratio{args.shift_ratio}_{timestamp}.txt"
    print(f"Saving experiment statistics to: {stats_file}")

    # Run with continuous domain shift
    model, priv_dataset = setup_experiment(args, use_shift=True)

    # Get domains list for visualization
    domains_list = priv_dataset.DOMAINS_LIST

    results = run_experiment_with_shift(model, priv_dataset, args, stats_file)

    # Close wandb
    if args.wandb:
        wandb.finish()

    print(f"\nExperiment complete! Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()
