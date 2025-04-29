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
from utils.continuous_training import continuous_domain_shift_training, ContinuousDomainShift
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


def run_experiment_with_shift(model, priv_dataset, args, stats_file):
    """Run experiment with continuous domain shift and save statistics"""
    print("Running experiment with continuous domain shift")
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

    # Store all statistics for detailed analysis
    all_stats = {
        'domain_shifts': [],
        'round_stats': [],
        'config': vars(args),
        'domains_list': domains_list,
    }

    # Get all data loaders for all domains
    domain_dataloaders = {}
    for domain in domains_list:
        # Get train and test loaders for this domain
        train_loaders, test_loaders = priv_dataset.get_data_loaders([domain])
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
            all_stats['domain_shifts'].append({
                'round': epoch_index,
                'distribution': domain_shifter.get_domain_distribution()
            })

            # Save domain shift information to stats file
            with open(stats_file, 'a') as f:
                f.write(f"===== DOMAIN SHIFT AT ROUND {epoch_index} =====\n")
                f.write("New domain distribution:\n")
                for domain, count in domain_shifter.get_domain_distribution().items():
                    f.write(f"  {domain}: {count} clients\n")
                f.write("\n")

        # Assign dataloader to each client based on their current domain
        current_train_loaders = []
        for client_id in range(args.parti_num):
            client_domain = domain_shifter.get_client_domain(client_id)
            # Get a dataloader from the client's current domain
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
        accs = global_evaluate(model, all_test_loaders, priv_dataset.SETTING, priv_dataset.NAME)

        # Calculate and track metrics
        mean_acc = round(np.mean(accs, axis=0), 3)
        mean_accs_list.append(mean_acc)

        # Record per-domain accuracies
        domain_accs = {}
        for i in range(len(accs)):
            if i in accs_dict:
                accs_dict[i].append(accs[i])
            else:
                accs_dict[i] = [accs[i]]
            domain_accs[i] = accs[i]

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
        round_stats = {
            'round': epoch_index,
            'mean_accuracy': mean_acc,
            'domain_accuracies': domain_accs,
            'domain_distribution': domain_shifter.get_domain_distribution(),
            'domain_shift_occurred': shift_occurred,
            'best_accuracy_so_far': best_acc
        }
        all_stats['round_stats'].append(round_stats)

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

    # Write final summary statistics
    with open(stats_file, 'a') as f:
        f.write("\n===== EXPERIMENT SUMMARY =====\n")
        f.write(f"Best Mean Accuracy: {best_acc:.2f}%\n")
        f.write(f"Best Domain Accuracies: {best_accs}\n")

        # Calculate stability metrics
        shift_rounds = [r for r in range(args.communication_epoch) if r > 0 and r % args.shift_frequency == 0]
        stability_metrics = []

        for shift_round in shift_rounds:
            if shift_round + 3 < args.communication_epoch:
                post_shift_variance = np.var(mean_accs_list[shift_round:shift_round + 3])
                stability_metrics.append(post_shift_variance)
                f.write(f"Accuracy variance after shift at round {shift_round}: {post_shift_variance:.4f}\n")

        if stability_metrics:
            avg_post_shift_variance = np.mean(stability_metrics)
            f.write(f"Average variance after shifts: {avg_post_shift_variance:.4f}\n")

        f.write("\nDomain distribution at the end of experiment:\n")
        for domain, count in domain_shifter.get_domain_distribution().items():
            f.write(f"  {domain}: {count} clients\n")

    # Also save all statistics as JSON for programmatic analysis
    json_stats_file = stats_file.replace('.txt', '.json')
    with open(json_stats_file, 'w') as f:
        # Convert numpy arrays and other non-serializable objects to lists
        serializable_stats = json.dumps(all_stats, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else vars(
            o) if hasattr(o, '__dict__') else str(o))
        f.write(serializable_stats)

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


def analyze_results(results, args, domains_list, stats_file):
    """Analyze and visualize experiment results"""
    if results is None:
        print("No results to analyze (baseline experiment)")
        return

    # Extract metrics
    accs_dict = results['accs_dict']
    mean_accs_list = results['mean_accs_list']
    domain_shift_history = results['domain_shift_history']

    # Create directory for visualizations if it doesn't exist
    viz_dir = "visualizations"
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)

    # Plot accuracy over time
    plt.figure(figsize=(12, 6))

    # Mean accuracy
    plt.plot(mean_accs_list, label='Mean Accuracy', linewidth=2, color='black')

    # Individual domain accuracies
    colors = plt.cm.tab10(np.linspace(0, 1, len(accs_dict)))
    for i, (domain_id, accs) in enumerate(accs_dict.items()):
        plt.plot(accs, label=f'Domain {domain_id} ({domains_list[int(domain_id)]})', alpha=0.7, color=colors[i])

    # Mark domain shift points
    for round_idx in range(args.communication_epoch):
        if round_idx > 0 and round_idx % args.shift_frequency == 0:
            plt.axvline(x=round_idx, color='red', linestyle='--', alpha=0.3)

    plt.xlabel('Communication Round')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Time with Domain Shifts')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save or display
    plt.tight_layout()
    viz_filename = f"{viz_dir}/{args.model}_{args.dataset}_accuracy.png"
    plt.savefig(viz_filename)

    # Update the stats file with visualization path
    with open(stats_file, 'a') as f:
        f.write(f"\nAccuracy visualization saved to: {viz_filename}\n")

    if args.wandb:
        wandb.log({"accuracy_plot": wandb.Image(plt)})

    # Domain shift visualization
    plt.figure(figsize=(12, 8))

    # Create a matrix to visualize domain shifts
    domain_to_id = {domain: i for i, domain in enumerate(domains_list)}

    shift_matrix = np.zeros((args.parti_num, args.communication_epoch))

    for client_id, domain_history in domain_shift_history.items():
        # Extend history to match communication epochs if needed
        full_history = domain_history.copy()
        while len(full_history) < args.communication_epoch:
            full_history.append(full_history[-1])

        # Fill matrix with domain IDs
        for round_idx, domain in enumerate(full_history[:args.communication_epoch]):
            shift_matrix[client_id, round_idx] = domain_to_id[domain]

    # Plot heatmap
    sns.heatmap(shift_matrix, cmap='tab10', cbar=False,
                xticklabels=10, yticklabels=True)

    # Create custom legend for domains
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=plt.cm.tab10(domain_to_id[domain] / len(domains_list)),
                             label=domain) for domain in domains_list]
    plt.legend(handles=legend_elements, bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.xlabel('Communication Round')
    plt.ylabel('Client ID')
    plt.title('Domain Shifts Over Time')

    # Save or display
    plt.tight_layout()
    shifts_viz_filename = f"{viz_dir}/{args.model}_{args.dataset}_domain_shifts.png"
    plt.savefig(shifts_viz_filename)

    # Update the stats file with visualization path
    with open(stats_file, 'a') as f:
        f.write(f"Domain shift visualization saved to: {shifts_viz_filename}\n")

    if args.wandb:
        wandb.log({"domain_shifts": wandb.Image(plt)})

    # Print final statistics
    print("\nFinal Results:")
    print(f"Best Mean Accuracy: {max(mean_accs_list):.2f}%")

    # Calculate stability metrics (variance after shifts)
    shift_rounds = [r for r in range(args.communication_epoch) if r > 0 and r % args.shift_frequency == 0]
    stability_metrics = []

    for shift_round in shift_rounds:
        if shift_round + 3 < args.communication_epoch:  # Need at least 3 rounds after shift
            # Calculate variance in accuracy for 3 rounds after shift
            post_shift_variance = np.var(mean_accs_list[shift_round:shift_round + 3])
            stability_metrics.append(post_shift_variance)

    if stability_metrics:
        avg_post_shift_variance = np.mean(stability_metrics)
        print(f"Average Variance After Shifts: {avg_post_shift_variance:.4f}")

        if args.wandb:
            wandb.log({"avg_post_shift_variance": avg_post_shift_variance})


def main():
    args = parse_args()

    # Create directory for stats files if it doesn't exist
    stats_dir = "experiment_stats"
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)

    # Create a timestamp for the experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a stats file for the experiment
    stats_file = f"{stats_dir}/{args.model}_{args.dataset}_shift{args.shift_frequency}_{timestamp}.txt"
    print(f"Saving experiment statistics to: {stats_file}")

    # Run with continuous domain shift
    model, priv_dataset = setup_experiment(args, use_shift=True)

    # Get domains list for visualization
    domains_list = priv_dataset.DOMAINS_LIST

    results = run_experiment_with_shift(model, priv_dataset, args, stats_file)

    # Analyze results
    analyze_results(results, args, domains_list, stats_file)

    # Close wandb
    if args.wandb:
        wandb.finish()

    # Optionally run baseline experiment (no domain shift)
    if args.run_baseline:
        print("\n" + "=" * 50)
        print("Running baseline experiment (static domains)")
        print("=" * 50 + "\n")

        # Create a stats file for the baseline experiment
        baseline_stats_file = f"{stats_dir}/{args.model}_{args.dataset}_baseline_{timestamp}.txt"
        print(f"Saving baseline statistics to: {baseline_stats_file}")

        with open(baseline_stats_file, 'w') as f:
            f.write("===== BASELINE EXPERIMENT (STATIC DOMAINS) =====\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Backbone: {args.backbone}\n")
            f.write(f"Participants: {args.parti_num}\n")
            f.write(f"Communication Epochs: {args.communication_epoch}\n")
            f.write(f"Local Epochs: {args.local_epoch}\n\n")
            f.write("Running standard training without domain shifts...\n")

        # Reset wandb
        if args.wandb:
            wandb.finish()

        # Run baseline with same settings but no shift
        model, priv_dataset = setup_experiment(args, use_shift=False)
        run_baseline_experiment(model, priv_dataset, args)

        with open(baseline_stats_file, 'a') as f:
            f.write("\nBaseline experiment completed.\n")
            f.write("Note: Detailed per-round statistics are not available for baseline experiments.\n")

        # Close wandb again
        if args.wandb:
            wandb.finish()

    print(f"\nExperiment complete! Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()