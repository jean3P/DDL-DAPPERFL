#!/usr/bin/env python3
"""
Domain Shift Impact Analysis for Federated Learning Experiments
Analyzes the impact of domain shifts on model accuracy across different domains
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import os
import re


def parse_text_log(filepath):
    """Parse the text log file to extract experiment data"""
    data = {
        'config': {},
        'rounds': [],
        'shifts': [],
        'summary': {},
        'domain_names': []
    }

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Parse configuration
    for i, line in enumerate(lines):
        if 'Model:' in line:
            data['config']['model'] = line.split(':')[1].strip()
        elif 'Dataset:' in line:
            data['config']['dataset'] = line.split(':')[1].strip()
        elif 'Pruning Strategy:' in line:
            data['config']['pruning_strategy'] = line.split(':')[1].strip()
        elif 'Shift Frequency:' in line:
            data['config']['shift_frequency'] = int(line.split(':')[1].strip())
        elif 'Shift Ratio:' in line:
            data['config']['shift_ratio'] = float(line.split(':')[1].strip())
        elif 'Communication Epochs:' in line:
            data['config']['num_rounds'] = int(line.split(':')[1].strip())
        elif 'Participants:' in line:
            data['config']['num_participants'] = int(line.split(':')[1].strip())

        # Parse round data
        if 'Round' in line and 'Mean Acc:' in line:
            # Ignore the template header line or anything that lacks a real round number
            m = re.search(r'\bRound\s+(\d+)', line)
            if not m:  # nothing like “Round 17” found – skip
                continue
            round_num = int(m.group(1))
            parts = line.split('|')
            mean_acc = float(parts[1].split(':')[1].strip().rstrip('%'))
            shifted = 'Yes' in parts[3]

            # Extract domain accuracies
            domain_accs_str = parts[2].split(':')[1].strip()
            domain_accs = [float(x) for x in domain_accs_str.strip('[]').split(', ')]

            data['rounds'].append({
                'round': round_num,
                'mean_accuracy': mean_acc,
                'domain_accuracies': domain_accs,
                'shifted': shifted
            })

        # Parse domain shift events
        if 'DOMAIN SHIFT AT ROUND' in line:
            round_num = int(line.split()[-2])
            data['shifts'].append(round_num)

        # Extract domain names from domain distribution
        if 'New domain distribution:' in line:
            i_start = lines.index(line) + 1
            while i_start < len(lines) and lines[i_start].strip() and ':' in lines[i_start]:
                domain_name = lines[i_start].split(':')[0].strip()
                if domain_name not in data['domain_names']:
                    data['domain_names'].append(domain_name)
                i_start += 1

    # Parse summary statistics
    for i, line in enumerate(lines):
        if 'Best Mean Accuracy:' in line:
            data['summary']['best_accuracy'] = float(line.split(':')[1].strip().rstrip('%'))
        elif 'Average Accuracy:' in line and 'Drop' not in line:
            data['summary']['avg_accuracy'] = float(line.split(':')[1].strip().rstrip('%'))
        elif 'Number of Domain Shifts:' in line:
            data['summary']['num_shifts'] = int(line.split(':')[1].strip())
        elif 'Average Accuracy Drop After Shift:' in line:
            data['summary']['avg_drop_after_shift'] = float(line.split(':')[1].strip())
        elif 'Final 10 Rounds Avg:' in line:
            data['summary']['final_10_avg'] = float(line.split(':')[1].strip().rstrip('%'))
        elif 'Accuracy Std Dev:' in line:
            data['summary']['accuracy_std'] = float(line.split(':')[1].strip())

    return data


def parse_json_log(filepath):
    """Parse the JSON log file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def load_baseline_results(filepath):
    """Load baseline results from a JSON file"""
    if filepath and os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None


def analyze_domain_shift_impact(text_data, json_data, baseline_data=None):
    """Analyze the impact of domain shifts on accuracy"""
    # Get domain names from JSON data
    domains = list(json_data['domains_list']) if 'domains_list' in json_data else []
    if not domains and text_data['domain_names']:
        domains = text_data['domain_names']

    analysis = {
        'pruning_strategy': text_data['config']['pruning_strategy'],
        'shift_frequency': text_data['config']['shift_frequency'],
        'shift_ratio': text_data['config']['shift_ratio'],
        'domains': domains,
        'total_rounds': text_data['config']['num_rounds']
    }

    # Get final domain accuracies (average of last 10 rounds)
    last_10_rounds = json_data['rounds'][-10:]
    domain_accs = defaultdict(list)

    for round_data in last_10_rounds:
        for domain, acc in round_data['accuracies'].items():
            domain_accs[domain].append(acc)

    analysis['final_domain_accuracies'] = {
        domain: np.mean(accs) for domain, accs in domain_accs.items()
    }

    # Calculate overall final accuracy
    all_final_accs = [acc for accs in domain_accs.values() for acc in accs]
    analysis['final_global_accuracy'] = np.mean(all_final_accs) if all_final_accs else 0

    # Get accuracy evolution per domain
    domain_evolution = defaultdict(list)
    for round_data in json_data['rounds']:
        for domain, acc in round_data['accuracies'].items():
            domain_evolution[domain].append(acc)

    analysis['domain_evolution'] = dict(domain_evolution)

    # Analyze accuracy drops after shifts
    drops = []
    recoveries = []

    for i, shift_round in enumerate(text_data['shifts']):
        if shift_round > 0 and shift_round < len(text_data['rounds']):
            pre_shift_acc = text_data['rounds'][shift_round - 1]['mean_accuracy']
            post_shift_acc = text_data['rounds'][shift_round]['mean_accuracy']
            drop = pre_shift_acc - post_shift_acc
            drops.append(drop)

            # Find recovery (rounds to reach pre-shift accuracy)
            recovery_rounds = 0
            for j in range(shift_round + 1, min(shift_round + 10, len(text_data['rounds']))):
                if text_data['rounds'][j]['mean_accuracy'] >= pre_shift_acc:
                    recovery_rounds = j - shift_round
                    break
            recoveries.append(recovery_rounds)

    analysis['accuracy_drops'] = {
        'mean': np.mean(drops) if drops else 0,
        'std': np.std(drops) if drops else 0,
        'max': max(drops) if drops else 0,
        'min': min(drops) if drops else 0,
        'all_drops': drops
    }

    analysis['recovery_rounds'] = {
        'mean': np.mean([r for r in recoveries if r > 0]) if any(r > 0 for r in recoveries) else 0,
        'all': recoveries
    }

    # Compare with baseline if provided
    if baseline_data:
        pruning_strategy = analysis['pruning_strategy']
        if pruning_strategy in baseline_data:
            baseline = baseline_data[pruning_strategy]
            analysis['baseline_comparison'] = {}

            for domain in analysis['domains']:
                if domain in baseline and domain in analysis['final_domain_accuracies']:
                    analysis['baseline_comparison'][domain] = {
                        'baseline': baseline[domain],
                        'with_shifts': analysis['final_domain_accuracies'][domain],
                        'improvement': analysis['final_domain_accuracies'][domain] - baseline[domain]
                    }

            # Global comparison
            if 'global' in baseline:
                analysis['baseline_comparison']['global'] = {
                    'baseline': baseline['global'],
                    'with_shifts': analysis['final_global_accuracy'],
                    'improvement': analysis['final_global_accuracy'] - baseline['global']
                }

    # Additional statistics
    all_accuracies = [r['mean_accuracy'] for r in text_data['rounds']]
    analysis['overall_stats'] = {
        'mean': np.mean(all_accuracies),
        'std': np.std(all_accuracies),
        'max': max(all_accuracies),
        'min': min(all_accuracies)
    }

    # Domain shift statistics
    if 'client_metrics' in json_data:
        domain_changes = []
        for client_id, metrics in json_data['client_metrics'].items():
            if 'shift_rounds' in metrics:
                domain_changes.append(len(metrics['shift_rounds']))

        analysis['client_domain_changes'] = {
            'mean': np.mean(domain_changes) if domain_changes else 0,
            'max': max(domain_changes) if domain_changes else 0,
            'min': min(domain_changes) if domain_changes else 0
        }

    return analysis


def plot_results(text_data, analysis, save_individual=False, output_dir='.'):
    """Create visualizations of the results"""
    # Determine subplot configuration based on whether we have baseline data
    has_baseline = 'baseline_comparison' in analysis

    if has_baseline:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    else:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Accuracy over rounds with shift markers
    rounds = [r['round'] for r in text_data['rounds']]
    accuracies = [r['mean_accuracy'] for r in text_data['rounds']]

    ax1.plot(rounds, accuracies, 'b-', linewidth=2, label='Mean Accuracy')

    # Add moving average
    window = 5
    if len(accuracies) >= window:
        moving_avg = np.convolve(accuracies, np.ones(window) / window, mode='valid')
        ax1.plot(rounds[window - 1:], moving_avg, 'g--', alpha=0.7, label=f'{window}-round Moving Avg')

    # Mark domain shifts
    for i, shift in enumerate(text_data['shifts']):
        ax1.axvline(x=shift, color='r', linestyle='--', alpha=0.5,
                    label='Domain Shift' if i == 0 else '')

    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title(
        f'Accuracy Evolution with Domain Shifts (Freq={analysis["shift_frequency"]}, Ratio={analysis["shift_ratio"]})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Domain-wise accuracy evolution
    if 'domain_evolution' in analysis:
        for domain, accs in analysis['domain_evolution'].items():
            ax2.plot(range(len(accs)), accs, label=domain.capitalize(), linewidth=2)

        # Mark shifts
        for shift in text_data['shifts'][:3]:  # Show first 3 shifts to avoid clutter
            ax2.axvline(x=shift, color='gray', linestyle=':', alpha=0.5)

        ax2.set_xlabel('Communication Round')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Domain-wise Accuracy Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # 3. Accuracy drops distribution
    if analysis['accuracy_drops']['all_drops']:
        drops = analysis['accuracy_drops']['all_drops']
        ax3.hist(drops, bins=min(15, len(drops)), edgecolor='black', alpha=0.7, color='salmon')
        ax3.axvline(x=analysis['accuracy_drops']['mean'], color='r', linestyle='--',
                    label=f'Mean: {analysis["accuracy_drops"]["mean"]:.2f}%')
        ax3.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_xlabel('Accuracy Drop (%)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Accuracy Drops After Domain Shifts')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # 4. Final accuracies or comparison with baseline
    if has_baseline and analysis['baseline_comparison']:
        domains = [d for d in analysis['domains'] if d in analysis['baseline_comparison']]
        domains_with_global = domains + ['global'] if 'global' in analysis['baseline_comparison'] else domains

        baseline_accs = [analysis['baseline_comparison'][d]['baseline'] for d in domains_with_global]
        shift_accs = [analysis['baseline_comparison'][d]['with_shifts'] for d in domains_with_global]

        x = np.arange(len(domains_with_global))
        width = 0.35

        ax4.bar(x - width / 2, baseline_accs, width, label=f'Baseline ({analysis["pruning_strategy"]})',
                color='lightcoral')
        ax4.bar(x + width / 2, shift_accs, width, label=f'With Shifts', color='lightblue')

        # Add improvement text
        for i, d in enumerate(domains_with_global):
            imp = analysis['baseline_comparison'][d]['improvement']
            ax4.text(i, max(baseline_accs[i], shift_accs[i]) + 1,
                     f'{imp:+.1f}%', ha='center', va='bottom', fontsize=9)

        ax4.set_xlabel('Domain')
        ax4.set_ylabel('Accuracy (%)')
        ax4.set_title('Accuracy Comparison: Baseline vs With Domain Shifts')
        ax4.set_xticks(x)
        ax4.set_xticklabels([d.capitalize() for d in domains_with_global])
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
    else:
        # Just show final accuracies
        domains = list(analysis['final_domain_accuracies'].keys())
        final_accs = list(analysis['final_domain_accuracies'].values())

        bars = ax4.bar(range(len(domains)), final_accs, color='lightblue', edgecolor='black')

        # Add value labels on bars
        for bar, acc in zip(bars, final_accs):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{acc:.1f}%', ha='center', va='bottom')

        ax4.set_xlabel('Domain')
        ax4.set_ylabel('Final Accuracy (%)')
        ax4.set_title(f'Final Domain Accuracies (Last 10 rounds average)')
        ax4.set_xticks(range(len(domains)))
        ax4.set_xticklabels([d.capitalize() for d in domains])
        ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save individual plots if requested
    if save_individual:
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(f'{output_dir}/plot_{i + 1}.png', bbox_inches=extent.expanded(1.2, 1.2), dpi=150)

    return fig


def print_analysis_report(analysis):
    """Print a comprehensive analysis report"""
    print("\n" + "=" * 80)
    print("DOMAIN SHIFT IMPACT ANALYSIS REPORT")
    print("=" * 80)

    print(f"\nExperiment Configuration:")
    print(f"  - Pruning Strategy: {analysis['pruning_strategy']}")
    print(f"  - Shift Frequency: Every {analysis['shift_frequency']} rounds")
    print(f"  - Shift Ratio: {analysis['shift_ratio']} ({analysis['shift_ratio'] * 100:.0f}% of clients shift)")
    print(f"  - Total Rounds: {analysis['total_rounds']}")

    print(f"\nFinal Domain Accuracies (Last 10 rounds average):")
    for domain in analysis['domains']:
        if domain in analysis['final_domain_accuracies']:
            acc = analysis['final_domain_accuracies'][domain]
            print(f"  - {domain.capitalize()}: {acc:.2f}%", end='')

            if 'baseline_comparison' in analysis and domain in analysis['baseline_comparison']:
                baseline = analysis['baseline_comparison'][domain]['baseline']
                imp = analysis['baseline_comparison'][domain]['improvement']
                print(f" (baseline: {baseline:.2f}%, change: {'+' if imp > 0 else ''}{imp:.2f}%)")
            else:
                print()

    print(f"\nGlobal Accuracy:")
    print(f"  - Final: {analysis['final_global_accuracy']:.2f}%")
    if 'baseline_comparison' in analysis and 'global' in analysis['baseline_comparison']:
        baseline = analysis['baseline_comparison']['global']['baseline']
        imp = analysis['baseline_comparison']['global']['improvement']
        print(f"  - Baseline: {baseline:.2f}%")
        print(f"  - Change: {'+' if imp > 0 else ''}{imp:.2f}%")

    print(f"\nDomain Shift Impact:")
    print(f"  - Total shifts: {len(analysis['accuracy_drops']['all_drops'])}")
    print(f"  - Average accuracy drop: {analysis['accuracy_drops']['mean']:.2f}% "
          f"(±{analysis['accuracy_drops']['std']:.2f}%)")
    print(f"  - Maximum drop: {analysis['accuracy_drops']['max']:.2f}%")
    print(f"  - Minimum drop: {analysis['accuracy_drops']['min']:.2f}%")
    print(f"  - Average recovery time: {analysis['recovery_rounds']['mean']:.2f} rounds")

    print(f"\nOverall Performance Statistics:")
    print(f"  - Mean accuracy: {analysis['overall_stats']['mean']:.2f}%")
    print(f"  - Std deviation: {analysis['overall_stats']['std']:.2f}%")
    print(f"  - Best accuracy: {analysis['overall_stats']['max']:.2f}%")
    print(f"  - Worst accuracy: {analysis['overall_stats']['min']:.2f}%")

    if 'client_domain_changes' in analysis:
        print(f"\nClient Domain Changes:")
        print(f"  - Average changes per client: {analysis['client_domain_changes']['mean']:.2f}")
        print(f"  - Maximum changes: {analysis['client_domain_changes']['max']}")
        print(f"  - Minimum changes: {analysis['client_domain_changes']['min']}")

    print("\nKey Findings:")

    # Overall impact
    if 'baseline_comparison' in analysis and 'global' in analysis['baseline_comparison']:
        global_imp = analysis['baseline_comparison']['global']['improvement']
        if global_imp > 0:
            print(f"  ✓ Domain shifts IMPROVED overall accuracy by {global_imp:.2f}%")
        else:
            print(f"  ✗ Domain shifts DECREASED overall accuracy by {abs(global_imp):.2f}%")

        # Domain-specific findings
        improvements = [(d, data['improvement']) for d, data in analysis['baseline_comparison'].items() if
                        d != 'global']
        best_domain = max(improvements, key=lambda x: x[1])
        worst_domain = min(improvements, key=lambda x: x[1])

        print(
            f"  - Best improvement: {best_domain[0].capitalize()} ({'+' if best_domain[1] > 0 else ''}{best_domain[1]:.2f}%)")
        print(
            f"  - Worst performance: {worst_domain[0].capitalize()} ({'+' if worst_domain[1] > 0 else ''}{worst_domain[1]:.2f}%)")

    # Stability analysis
    if analysis['accuracy_drops']['std'] < 2.0:
        print(f"  - Model shows good stability with low variance in drops ({analysis['accuracy_drops']['std']:.2f}%)")
    else:
        print(f"  - Model shows high variance in accuracy drops ({analysis['accuracy_drops']['std']:.2f}%)")

    if analysis['recovery_rounds']['mean'] < 3:
        print(f"  - Fast recovery from domain shifts (avg: {analysis['recovery_rounds']['mean']:.1f} rounds)")
    else:
        print(f"  - Slow recovery from domain shifts (avg: {analysis['recovery_rounds']['mean']:.1f} rounds)")

    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Analyze domain shift impact in federated learning experiments')
    parser.add_argument('--log-txt', required=True, help='Path to text log file')
    parser.add_argument('--log-json', required=True, help='Path to JSON log file')
    parser.add_argument('--baseline-json', help='Path to baseline results JSON file')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to file')
    parser.add_argument('--save-individual-plots', action='store_true', help='Save each plot separately')
    parser.add_argument('--output-dir', default='.', help='Directory to save outputs')
    parser.add_argument('--no-display', action='store_true', help='Do not display plots (useful for servers)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if args.save_plots or args.save_individual_plots:
        os.makedirs(args.output_dir, exist_ok=True)

    # Parse log files
    print("Parsing log files...")
    try:
        text_data = parse_text_log(args.log_txt)
        json_data = parse_json_log(args.log_json)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease check that the file paths are correct.")
        print(f"Looking for:")
        print(f"  - Text log: {args.log_txt}")
        print(f"  - JSON log: {args.log_json}")
        return
    except Exception as e:
        print(f"Error parsing log files: {e}")
        return

    # Load baseline results if provided
    baseline_data = None
    if args.baseline_json:
        baseline_data = load_baseline_results(args.baseline_json)
        if baseline_data:
            print(f"Loaded baseline results from {args.baseline_json}")

    # Analyze domain shift impact
    print("Analyzing domain shift impact...")
    analysis = analyze_domain_shift_impact(text_data, json_data, baseline_data)

    # Print analysis report
    print_analysis_report(analysis)

    # Create visualizations
    print("\nGenerating visualizations...")

    if not args.no_display or args.save_plots:
        fig = plot_results(text_data, analysis,
                           save_individual=args.save_individual_plots,
                           output_dir=args.output_dir)

        if args.save_plots:
            plot_path = f"{args.output_dir}/domain_shift_analysis.png"
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to: {plot_path}")

        if not args.no_display:
            plt.show()
        else:
            plt.close(fig)

    # Save analysis results to JSON
    output_path = f"{args.output_dir}/domain_shift_analysis_results.json"
    with open(output_path, 'w') as f:
        # Convert numpy values to Python types for JSON serialization
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(v) for v in obj]
            else:
                return obj

        json_safe_analysis = convert_to_json_serializable(analysis)
        json.dump(json_safe_analysis, f, indent=2)

    print(f"\nAnalysis results saved to: {output_path}")

    # Save baseline comparison if available
    if 'baseline_comparison' in analysis:
        baseline_output = f"{args.output_dir}/baseline_comparison.json"
        with open(baseline_output, 'w') as f:
            json.dump(analysis['baseline_comparison'], f, indent=2)
        print(f"Baseline comparison saved to: {baseline_output}")


if __name__ == "__main__":
    main()
