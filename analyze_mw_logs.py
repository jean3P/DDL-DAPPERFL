# analyze_mw_logs.py
# Script to analyze MW fairness logs after training

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import argparse


def load_experiment_data(log_dir):
    """Load all data files from an experiment directory"""
    data = {}

    # Load config
    config_files = glob(os.path.join(log_dir, "*_config.json"))
    if config_files:
        with open(config_files[0], 'r') as f:
            data['config'] = json.load(f)

    # Load metrics
    metrics_files = glob(os.path.join(log_dir, "*_metrics.csv"))
    if metrics_files:
        data['metrics'] = pd.read_csv(metrics_files[0])

    # Load lambda values
    lambda_files = glob(os.path.join(log_dir, "*_lambda.csv"))
    if lambda_files:
        data['lambda'] = pd.read_csv(lambda_files[0])

    # Load custom metrics
    custom_files = glob(os.path.join(log_dir, "*_custom.csv"))
    if custom_files:
        data['custom'] = pd.read_csv(custom_files[0])

    # Load client mapping
    mapping_files = glob(os.path.join(log_dir, "*_client_mapping.json"))
    if mapping_files:
        with open(mapping_files[0], 'r') as f:
            data['mapping'] = json.load(f)

    return data


def compare_experiments(log_dirs, labels=None):
    """Compare multiple experiments"""
    if labels is None:
        labels = [os.path.basename(d) for d in log_dirs]

    plt.figure(figsize=(15, 10))

    # Subplot 1: Accuracy comparison
    plt.subplot(2, 3, 1)
    for log_dir, label in zip(log_dirs, labels):
        data = load_experiment_data(log_dir)
        if 'metrics' in data:
            df = data['metrics'][data['metrics']['round'] != 'final']
            df['round'] = pd.to_numeric(df['round'])
            plt.plot(df['round'], df['mean_acc'], label=label, linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Mean Accuracy (%)')
    plt.title('Accuracy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: TPRD comparison
    plt.subplot(2, 3, 2)
    for log_dir, label in zip(log_dirs, labels):
        data = load_experiment_data(log_dir)
        if 'metrics' in data:
            df = data['metrics'][data['metrics']['round'] != 'final']
            df['round'] = pd.to_numeric(df['round'])
            plt.plot(df['round'], df['tprd'], label=label, linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('TPRD')
    plt.title('TPR Discrepancy Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Final metrics bar chart
    plt.subplot(2, 3, 3)
    final_metrics = {'Accuracy': [], 'TPRD': [], 'WTPR': []}
    for log_dir in log_dirs:
        data = load_experiment_data(log_dir)
        if 'metrics' in data:
            final_row = data['metrics'][data['metrics']['round'] == 'final']
            if len(final_row) > 0:
                final_metrics['Accuracy'].append(float(final_row['mean_acc'].values[0]))
                final_metrics['TPRD'].append(float(final_row['tprd'].values[0]))
                final_metrics['WTPR'].append(float(final_row['wtpr'].values[0]))
            else:
                # Use last row if no final row
                last_row = data['metrics'].iloc[-1]
                final_metrics['Accuracy'].append(float(last_row['mean_acc']))
                final_metrics['TPRD'].append(float(last_row['tprd']))
                final_metrics['WTPR'].append(float(last_row['wtpr']))

    x = range(len(labels))
    width = 0.25
    plt.bar([i - width for i in x], final_metrics['Accuracy'], width, label='Accuracy', alpha=0.8)
    plt.bar(x, [v * 100 for v in final_metrics['TPRD']], width, label='TPRD×100', alpha=0.8)
    plt.bar([i + width for i in x], final_metrics['WTPR'], width, label='WTPR', alpha=0.8)
    plt.xticks(x, labels, rotation=45)
    plt.ylabel('Value')
    plt.title('Final Metrics Comparison')
    plt.legend()

    # Subplot 4: Lambda evolution (if available)
    plt.subplot(2, 3, 4)
    has_lambda = False
    for log_dir, label in zip(log_dirs, labels):
        data = load_experiment_data(log_dir)
        if 'lambda' in data:
            has_lambda = True
            df = data['lambda'][data['lambda']['round'] != 'final']
            df['round'] = pd.to_numeric(df['round'])
            for col in ['lambda_0', 'lambda_1']:
                if col in df.columns:
                    plt.plot(df['round'], df[col], label=f'{label} - {col}', linewidth=2)
    if has_lambda:
        plt.xlabel('Round')
        plt.ylabel('Lambda Value')
        plt.title('Lambda Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.text(0.5, 0.5, 'No lambda data available', ha='center', va='center')

    # Subplot 5: Group performance heatmap
    plt.subplot(2, 3, 5)
    group_data = []
    exp_labels = []
    for log_dir, label in zip(log_dirs, labels):
        data = load_experiment_data(log_dir)
        if 'metrics' in data:
            final_row = data['metrics'][data['metrics']['round'] == 'final']
            if len(final_row) == 0:
                final_row = data['metrics'].iloc[-1:]

            group_tprs = []
            for g in range(5):
                col = f'group_{g}_mean_tpr'
                if col in final_row.columns:
                    val = final_row[col].values[0]
                    if pd.notna(val) and val > 0:
                        group_tprs.append(val)

            if group_tprs:
                group_data.append(group_tprs)
                exp_labels.append(label)

    if group_data:
        # Pad arrays to same length
        max_groups = max(len(g) for g in group_data)
        padded_data = []
        for g in group_data:
            padded = g + [0] * (max_groups - len(g))
            padded_data.append(padded)

        sns.heatmap(padded_data,
                    xticklabels=[f'Group {i}' for i in range(max_groups)],
                    yticklabels=exp_labels,
                    annot=True, fmt='.1f', cmap='YlOrRd',
                    cbar_kws={'label': 'Mean TPR (%)'})
        plt.title('Group Performance Heatmap')

    # Subplot 6: Convergence analysis
    plt.subplot(2, 3, 6)
    for log_dir, label in zip(log_dirs, labels):
        data = load_experiment_data(log_dir)
        if 'metrics' in data:
            df = data['metrics'][data['metrics']['round'] != 'final']
            df['round'] = pd.to_numeric(df['round'])

            # Calculate rolling average of TPRD
            window = min(10, len(df) // 4)
            if window > 1:
                df['tprd_smooth'] = df['tprd'].rolling(window=window, center=True).mean()
                plt.plot(df['round'], df['tprd_smooth'], label=f'{label} (smoothed)', linewidth=2)

    plt.xlabel('Round')
    plt.ylabel('TPRD (Smoothed)')
    plt.title('Fairness Convergence Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('mw_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_single_experiment(log_dir):
    """Detailed analysis of a single experiment"""
    data = load_experiment_data(log_dir)

    print("=" * 60)
    print(f"MW Fairness Analysis: {os.path.basename(log_dir)}")
    print("=" * 60)

    # Print configuration
    if 'config' in data:
        print("\nExperiment Configuration:")
        important_keys = ['model', 'dataset', 'communication_epoch', 'parti_num',
                          'batch_size', 'lr', 'group_fairness', 'num_groups']
        for key in important_keys:
            if key in data['config']:
                print(f"  {key}: {data['config'][key]}")

    # Print summary statistics
    if 'metrics' in data:
        df = data['metrics']
        numeric_df = df[df['round'] != 'final']
        numeric_df['round'] = pd.to_numeric(numeric_df['round'])

        print("\nPerformance Summary:")
        print(f"  Total rounds: {len(numeric_df)}")

        # Get final or last row
        final_row = df[df['round'] == 'final']
        if len(final_row) == 0:
            final_row = df.iloc[-1:]

        print(f"  Final accuracy: {float(final_row['mean_acc'].values[0]):.2f}%")
        print(f"  Final TPRD: {float(final_row['tprd'].values[0]):.4f}")
        print(f"  Final WTPR: {float(final_row['wtpr'].values[0]):.2f}%")
        print(f"  Final BTPR: {float(final_row['btpr'].values[0]):.2f}%")

        # Calculate improvements
        if len(numeric_df) > 1:
            acc_improvement = float(numeric_df['mean_acc'].iloc[-1] - numeric_df['mean_acc'].iloc[0])
            tprd_improvement = float(numeric_df['tprd'].iloc[0] - numeric_df['tprd'].iloc[-1])
            print(f"\n  Accuracy improvement: {acc_improvement:+.2f}%")
            print(f"  TPRD reduction: {tprd_improvement:.4f}")

        # Group analysis
        print("\nGroup Performance:")
        for g in range(5):
            mean_col = f'group_{g}_mean_tpr'
            count_col = f'group_{g}_count'
            if mean_col in final_row.columns:
                mean_tpr = final_row[mean_col].values[0]
                count = final_row[count_col].values[0]
                if pd.notna(mean_tpr) and mean_tpr > 0:
                    print(f"  Group {g}: {mean_tpr:.2f}% (n={int(count)} clients)")

    # Client mapping analysis
    if 'mapping' in data:
        print("\nClient Distribution:")
        client_groups = data['mapping']['client_groups']
        group_counts = {}
        for client, group in client_groups.items():
            group_counts[group] = group_counts.get(group, 0) + 1

        for group, count in sorted(group_counts.items()):
            print(f"  Group {group}: {count} clients")

        if 'noise_variances' in data['mapping'] and data['mapping']['noise_variances']:
            print("\nNoise Distribution:")
            noise_vars = data['mapping']['noise_variances']
            for client, variance in noise_vars.items():
                if variance > 0:
                    print(f"  Client {client}: σ² = {variance:.4f}")

    # Generate plots
    from utils.mw_logger import MWLogger

    # Create a dummy logger to use its plotting function
    class DummyArgs:
        def __init__(self, model, dataset):
            self.model = model
            self.dataset = dataset

    if 'config' in data:
        dummy_args = DummyArgs(data['config'].get('model', 'unknown'),
                               data['config'].get('dataset', 'unknown'))
        logger = MWLogger(dummy_args)
        logger.save_dir = log_dir
        logger.base_name = os.path.basename(glob(os.path.join(log_dir, "*_metrics.csv"))[0]).replace('_metrics.csv', '')
        logger.metrics_file = glob(os.path.join(log_dir, "*_metrics.csv"))[0]

        logger.plot_metrics(save_plots=True)


def main():
    parser = argparse.ArgumentParser(description='Analyze MW fairness logs')
    parser.add_argument('--log_dirs', nargs='+', required=True,
                        help='Directories containing MW logs to analyze')
    parser.add_argument('--labels', nargs='+',
                        help='Labels for each experiment (optional)')
    parser.add_argument('--single', action='store_true',
                        help='Perform detailed single experiment analysis')

    args = parser.parse_args()

    if args.single:
        for log_dir in args.log_dirs:
            analyze_single_experiment(log_dir)
    else:
        compare_experiments(args.log_dirs, args.labels)


if __name__ == '__main__':
    main()

# Example usage:
# python analyze_mw_logs.py --log_dirs mw_logs/mwfair_fl_digits_20250124_143022 --single
# python analyze_mw_logs.py --log_dirs exp1/mw_logs exp2/mw_logs exp3/mw_logs --labels "Baseline" "MW-Fair" "DapperFL-MW"