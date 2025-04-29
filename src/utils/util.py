# src/utils/util.py

import os
import numpy as np

import torch
import wandb

from utils.continuous_training import global_evaluate


def create_if_not_exists(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def save_networks(model, communication_idx):
    nets_list = model.nets_list
    model_name = model.NAME

    checkpoint_path = model.checkpoint_path
    model_path = os.path.join(checkpoint_path, model_name)
    model_para_path = os.path.join(model_path, 'para')
    create_if_not_exists(model_para_path)
    for net_idx, network in enumerate(nets_list):
        each_network_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(net_idx) + '.ckpt')
        torch.save(network.state_dict(), each_network_path)


def save_protos(model, communication_idx):
    model_name = model.NAME

    checkpoint_path = model.checkpoint_path
    model_path = os.path.join(checkpoint_path, model_name)
    model_para_path = os.path.join(model_path, 'protos')
    create_if_not_exists(model_para_path)

    for i in range(len(model.global_protos_all)):
        label = i
        protos = torch.cat(model.global_protos_all[i], dim=0).cpu().numpy()
        save_path = os.path.join(model_para_path, str(communication_idx) + '_' + str(label) + '.npy')
        np.save(save_path, protos)


def run_baseline_experiment(model, priv_dataset, args, baseline_stats_file=None):
    """Run baseline experiment without domain shift and save basic statistics"""
    print("Running baseline experiment (static domains)")

    # Get domains list
    domains_list = priv_dataset.DOMAINS_LIST

    # Track experiment progress
    best_acc = 0
    best_accs = []

    # Initialize model
    if hasattr(model, 'ini'):
        model.ini()

    # Get train and test loaders without domain shifts (using default distribution)
    train_loaders, test_loaders = priv_dataset.get_data_loaders()

    # Main training loop
    for epoch_index in range(args.communication_epoch):
        model.epoch_index = epoch_index

        # Set train loaders and perform local updates
        model.trainloaders = train_loaders
        if hasattr(model, 'loc_update'):
            epoch_loc_loss_dict = model.loc_update(train_loaders)

        # Evaluate on all domains
        accs = global_evaluate(model, test_loaders, priv_dataset.SETTING, priv_dataset.NAME)

        # Calculate mean accuracy
        mean_acc = round(np.mean(accs, axis=0), 3)

        # Update best accuracy
        is_best = False
        if mean_acc > best_acc:
            best_acc = mean_acc
            is_best = True

        if len(best_accs) == 0:
            best_accs = accs.copy()
        else:
            for i in range(len(accs)):
                if accs[i] > best_accs[i]:
                    best_accs[i] = accs[i]

        # Log metrics to wandb if available
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

        # Log to statistics file if provided
        if baseline_stats_file:
            with open(baseline_stats_file, 'a') as f:
                f.write(f"Round {epoch_index} | Mean Acc: {mean_acc:.2f}% | Domain Accs: {accs}\n")
                if is_best:
                    f.write(f"  ★ New best accuracy: {mean_acc:.2f}%\n")

        # Print progress
        print(f'Round: {epoch_index}, Method: {model.args.model}, '
              f'Mean_Acc: {mean_acc}, Best_Acc: {best_acc}')
        print(f'Domain_Acc: {accs}, Domain_BestAcc: {best_accs}')

    # Write final summary to stats file if provided
    if baseline_stats_file:
        with open(baseline_stats_file, 'a') as f:
            f.write("\n===== BASELINE EXPERIMENT SUMMARY =====\n")
            f.write(f"Best Mean Accuracy: {best_acc:.2f}%\n")
            f.write(f"Best Domain Accuracies: {best_accs}\n")

    return {
        'best_acc': best_acc,
        'best_accs': best_accs
    }
