# src/utils/training.py

import copy

import torch
from argparse import Namespace
from models.utils.federated_model import FederatedModel
from datasets.utils.federated_dataset import FederatedDataset
from typing import Tuple
from torch.utils.data import DataLoader
import numpy as np
from .logger import CsvWriter
from collections import Counter
from sklearn.metrics import recall_score
import wandb


def global_evaluate(model: FederatedModel, test_dl: DataLoader, setting: str, name: str) -> Tuple[list, list]:
    accs = []
    net = model.global_net
    status = net.training
    net.eval()
    for j, dl in enumerate(test_dl):
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
        top5acc = round(100 * top5 / total, 2)
        accs.append(top1acc)
    net.train(status)
    return accs


def global_evaluate_tpr(model: FederatedModel,
                        test_loaders: list[DataLoader]
                        ) -> list[float]:
    """
    Calculate macro-recall (TPR) per client.
    test_loaders is a list per client.
    """
    tpr_per_client = []
    net = model.global_net
    status = net.training
    net.eval()

    for dl in test_loaders:
        all_preds, all_labels = [], []
        for batch_idx, (images, labels) in enumerate(dl):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = net(images) if model.NAME != 'nefl' else net(images)[0]
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        tpr = recall_score(y_true, y_pred, average='macro')
        tpr_per_client.append(tpr)

    net.train(status)
    return tpr_per_client

def local_evaluate_tpr(nets_list: list[torch.nn.Module],
                       test_loaders_per_client: list[DataLoader],
                       device: torch.device,
                       is_nefl: bool = False
                       ) -> list[float]:
    tpr_per_client = []
    for net, dl in zip(nets_list, test_loaders_per_client):
        net.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in dl:
                images, labels = images.to(device), labels.to(device)
                outputs = (net(images)[0] if is_nefl else net(images))
                preds = outputs.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        tpr = recall_score(np.array(all_labels),
                           np.array(all_preds),
                           average='macro')
        tpr_per_client.append(tpr)
        net.train()
    return tpr_per_client


def local_evaluate(model: FederatedModel, test_dl: DataLoader, domains_list: list, selected_domain_list: list,
                   setting: str, name: str) -> list:
    all_accs = {}
    for i, net in enumerate(model.nets_list):
        status = net.training
        net.eval()
        domain = selected_domain_list[i]
        domain_index = domains_list.index(domain)
        if domain_index not in all_accs.keys():
            all_accs[domain_index] = []
        correct, total, top1, top5 = 0.0, 0.0, 0.0, 0.0
        for batch_idx, (images, labels) in enumerate(test_dl[domain_index]):
            with torch.no_grad():
                images, labels = images.to(model.device), labels.to(model.device)
                outputs = net(images)
                _, max5 = torch.topk(outputs, 5, dim=-1)
                labels = labels.view(-1, 1)
                top1 += (labels == max5[:, 0:1]).sum().item()
                top5 += (labels == max5).sum().item()
                total += labels.size(0)
        top1acc = round(100 * top1 / total, 2)
        all_accs[domain_index].append(top1acc)
        net.train(status)

    avg_accs = []
    for i in range(len(all_accs)):
        avg_acc = round(sum(all_accs[i]) / len(all_accs[i]), 2)
        avg_accs.append(avg_acc)
    return avg_accs


def train(model: FederatedModel, private_dataset: FederatedDataset,
          args: Namespace) -> None:
    if args.csv_log:
        csv_writer = CsvWriter(args, private_dataset)

    model.N_CLASS = private_dataset.N_CLASS
    domains_list = private_dataset.DOMAINS_LIST
    domains_len = len(domains_list)

    if args.rand_dataset:
        max_num = 10
        is_ok = False

        while not is_ok:
            if model.args.dataset == 'fl_officecaltech':
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True,
                                                        p=None)
                selected_domain_list = list(selected_domain_list) + domains_list
            elif model.args.dataset == 'fl_digits':
                # selected_domain_list = np.random.choice(domains_list, size=args.parti_num, replace=True, p=None)
                selected_domain_list = np.random.choice(domains_list, size=args.parti_num - domains_len, replace=True,
                                                        p=None)
                selected_domain_list = list(selected_domain_list) + domains_list

            result = dict(Counter(selected_domain_list))

            for k in result:
                if result[k] > max_num:
                    is_ok = False
                    break
            else:
                is_ok = True

    else:
        selected_domain_dict = {'mnist': 6, 'usps': 4, 'svhn': 3, 'syn': 7}

        selected_domain_list = []
        for k in selected_domain_dict:
            domain_num = selected_domain_dict[k]
            for i in range(domain_num):
                selected_domain_list.append(k)

        selected_domain_list = np.random.permutation(selected_domain_list)

        result = Counter(selected_domain_list)
    print(result)

    print(selected_domain_list)
    pri_train_loaders, test_loaders = private_dataset.get_data_loaders(selected_domain_list)
    # To calculate TPR
    test_loaders_per_client = [
        test_loaders[private_dataset.DOMAINS_LIST.index(dom)]
        for dom in selected_domain_list
    ]
    model.trainloaders = pri_train_loaders
    if hasattr(model, 'ini'):
        model.ini()

    accs_dict = {}
    mean_accs_list = []
    best_acc = 0
    best_accs = []

    noise_clients = getattr(args, 'noise_clients', None)
    if isinstance(noise_clients, str):
        noise_clients = [int(x) for x in noise_clients.split(',') if x.strip().isdigit()]

    Epoch = args.communication_epoch
    for epoch_index in range(Epoch):
        model.epoch_index = epoch_index

        if hasattr(model, 'loc_update'):
            if noise_clients:
                total_clients = list(range(model.args.parti_num))
                online_clients = model.random_state.choice(
                    total_clients, model.online_num, replace=False
                ).tolist()
                model.online_clients = online_clients
                for i in online_clients:
                    group_weights = (model.compute_group_weights()
                                     if model.use_group_fairness else None)
                    model._train_net(i,
                                     model.nets_list[i],
                                     pri_train_loaders[i],
                                     group_weights)

                local_tprs = local_evaluate_tpr(
                    model.nets_list,
                    test_loaders_per_client,
                    model.device,
                    is_nefl=(model.NAME == 'nefl')
                )
                formatted = [f"{tpr:.3f}" for tpr in local_tprs]
                print(f"Round {epoch_index} – LOCAL TPR pre-agg per client: {formatted}")
                model.aggregate_nets(None)
            else:
                epoch_loc_loss_dict = model.loc_update(pri_train_loaders)


        if args.model in ['localtest']:
            accs = local_evaluate(model, test_loaders, domains_list, selected_domain_list, private_dataset.SETTING,
                                  private_dataset.NAME)
            model.aggregate_nets()
        else:
            accs = global_evaluate(model, test_loaders, private_dataset.SETTING, private_dataset.NAME)
            tpr_list = global_evaluate_tpr(model, test_loaders_per_client)
            tpr_summary = ", ".join(f"{idx}:{tpr:.3f}"
                                    for idx, tpr in enumerate(tpr_list))
            print(f"Round {epoch_index} – TPR per Domain: {tpr_summary}")

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

        if args.wandb:
            wandb.log({"Best_Acc": best_acc, "Mean_Acc": mean_acc, "round": epoch_index})
            if len(best_accs) == 0:
                best_accs = copy.deepcopy(accs)
            for i in range(len(accs)):
                name = "Domain" + str(i)
                wandb.log({name + "_Acc": accs[i], name + "_BestAcc": best_accs[i], "round": epoch_index})

        print('Round:', str(epoch_index), 'Method:', model.args.model,
              'Mean_Acc:', str(mean_acc), 'Best_Acc:', str(best_acc))
        print('Domain_Acc:', accs, 'Domain_BestAcc:', best_accs)

    if args.csv_log:
        csv_writer.write_acc(accs_dict, mean_accs_list)

    # Log group fairness metrics if using MW algorithm
    if hasattr(args, 'group_fairness') and args.group_fairness:
        # Calculate and log group fairness metrics
        tpr_list = global_evaluate_tpr(model, test_loaders_per_client)

        # Calculate TPRD (TPR Discrepancy)
        tpr_max = max(tpr_list)
        tpr_min = min(tpr_list)
        tpr_discrepancy = tpr_max - tpr_min

        # Calculate TPRSD (TPR Standard Deviation)
        tpr_std = np.std(tpr_list)

        # Print final fairness metrics
        print(f"Final Group Fairness Metrics:")
        print(f"  TPRD (TPR Discrepancy): {tpr_discrepancy:.4f}")
        print(f"  TPRSD (TPR Standard Deviation): {tpr_std:.4f}")
        print(f"  WTPR (Worst-case TPR): {tpr_min:.4f}")
        print(f"  BTPR (Best-case TPR): {tpr_max:.4f}")

        if args.wandb:
            wandb.log({
                "Final_TPRD": tpr_discrepancy,
                "Final_TPRSD": tpr_std,
                "Final_WTPR": tpr_min,
                "Final_BTPR": tpr_max
            })

    # Save the final model if requested
    if hasattr(args, 'save_model') and args.save_model:
        save_path = os.path.join(
            checkpoint_path(),
            args.dataset,
            args.model,
            f"final_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        )
        torch.save(model.global_net.state_dict(), save_path)
        print(f"Final model saved to {save_path}")

    return {
        'accs_dict': accs_dict,
        'mean_accs_list': mean_accs_list,
        'best_acc': best_acc,
        'best_accs': best_accs,
        'final_tpr_list': tpr_list if 'tpr_list' in locals() else None
    }

