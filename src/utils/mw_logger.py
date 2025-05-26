# utils/mw_logger.py

import os
import json
import csv
from datetime import datetime


class MWLogger:
    def __init__(self, args, save_dir="mw_logs"):
        self.args = args
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Create unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base_name = f"{args.model}_{args.dataset}_{timestamp}"

        # Initialize log files
        self.metrics_file = os.path.join(save_dir, f"{self.base_name}_metrics.csv")
        self.lambda_file = os.path.join(save_dir, f"{self.base_name}_lambda.csv")
        self.config_file = os.path.join(save_dir, f"{self.base_name}_config.json")

        # Save configuration
        config_dict = {}
        for key, value in vars(args).items():
            try:
                # Try to serialize the value
                json.dumps(value)
                config_dict[key] = value
            except (TypeError, ValueError):
                # If not serializable, convert to string
                config_dict[key] = str(value)

        with open(self.config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Initialize CSV headers
        self._init_metrics_csv()
        self._init_lambda_csv()

    def _init_metrics_csv(self):
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Extended header to support more than 2 groups if needed
            headers = ['round', 'mean_acc', 'tprd', 'wtpr', 'btpr', 'tprsd']

            # Add group-specific headers (supporting up to 5 groups)
            for g in range(5):
                headers.extend([f'group_{g}_mean_tpr', f'group_{g}_min_tpr', f'group_{g}_count'])

            writer.writerow(headers)

    def _init_lambda_csv(self):
        with open(self.lambda_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Support up to 5 groups for lambda values
            headers = ['round'] + [f'lambda_{g}' for g in range(5)]
            writer.writerow(headers)

    def log_metrics(self, round_idx, mean_acc, mw_metrics):
        """Log MW metrics for a round"""
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                round_idx,
                mean_acc,
                mw_metrics.get('TPRD', 0),
                mw_metrics.get('WTPR', 0),
                mw_metrics.get('BTPR', 0),
                mw_metrics.get('TPRSD', 0)
            ]

            # Add group-specific metrics (up to 5 groups)
            for g in range(5):
                row.extend([
                    mw_metrics.get(f'group_{g}_mean_tpr', 0),
                    mw_metrics.get(f'group_{g}_min_tpr', 0),
                    mw_metrics.get(f'group_{g}_count', 0)
                ])

            writer.writerow(row)

    def log_lambda_values(self, round_idx, lambda_dict):
        """Log lambda values for a round"""
        with open(self.lambda_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [round_idx]

            # Add lambda values for up to 5 groups
            for g in range(5):
                row.append(lambda_dict.get(g, 0))

            writer.writerow(row)

    def log_custom_metrics(self, round_idx, metrics_dict):
        """Log additional custom metrics to a separate file"""
        custom_file = os.path.join(self.save_dir, f"{self.base_name}_custom.csv")

        # Create header if file doesn't exist
        if not os.path.exists(custom_file):
            with open(custom_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['round'] + list(metrics_dict.keys()))

        # Write metrics
        with open(custom_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([round_idx] + list(metrics_dict.values()))

    def log_client_mapping(self, client_groups, noise_variances=None):
        """Log client-to-group mapping and noise information"""
        mapping_file = os.path.join(self.save_dir, f"{self.base_name}_client_mapping.json")

        mapping = {
            'client_groups': client_groups,
            'noise_variances': noise_variances if noise_variances else {}
        }

        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)

    def get_summary_stats(self):
        """Read metrics file and return summary statistics"""
        try:
            import pandas as pd

            df = pd.read_csv(self.metrics_file)

            # Convert round column to string to handle 'final' value
            df['round'] = df['round'].astype(str)

            # Filter out 'final' row for numeric calculations
            numeric_df = df[df['round'] != 'final'].copy()
            numeric_df['round'] = pd.to_numeric(numeric_df['round'])

            # Get final row data
            final_row = df[df['round'] == 'final']
            has_final = len(final_row) > 0

            summary = {
                'total_rounds': len(numeric_df),
                'final_mean_acc': float(final_row['mean_acc'].values[0]) if has_final else float(
                    numeric_df['mean_acc'].iloc[-1]) if len(numeric_df) > 0 else 0,
                'final_tprd': float(final_row['tprd'].values[0]) if has_final else float(
                    numeric_df['tprd'].iloc[-1]) if len(numeric_df) > 0 else 0,
                'min_tprd': float(numeric_df['tprd'].min()) if len(numeric_df) > 0 else 0,
                'avg_tprd': float(numeric_df['tprd'].mean()) if len(numeric_df) > 0 else 0,
                'tprd_improvement': float(numeric_df['tprd'].iloc[0] - numeric_df['tprd'].iloc[-1]) if len(
                    numeric_df) > 1 else 0
            }

            return summary
        except ImportError:
            # Fallback without pandas
            try:
                with open(self.metrics_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) <= 1:  # Only header
                        return None

                    # Parse CSV manually
                    header = lines[0].strip().split(',')
                    round_idx = header.index('round')
                    mean_acc_idx = header.index('mean_acc')
                    tprd_idx = header.index('tprd')

                    numeric_rows = []
                    final_row = None

                    for line in lines[1:]:
                        values = line.strip().split(',')
                        if values[round_idx] == 'final':
                            final_row = values
                        else:
                            numeric_rows.append(values)

                    if not numeric_rows:
                        return None

                    tprd_values = [float(row[tprd_idx]) for row in numeric_rows]

                    summary = {
                        'total_rounds': len(numeric_rows),
                        'final_mean_acc': float(final_row[mean_acc_idx]) if final_row else float(
                            numeric_rows[-1][mean_acc_idx]),
                        'final_tprd': float(final_row[tprd_idx]) if final_row else float(numeric_rows[-1][tprd_idx]),
                        'min_tprd': min(tprd_values),
                        'avg_tprd': sum(tprd_values) / len(tprd_values),
                        'tprd_improvement': tprd_values[0] - tprd_values[-1] if len(tprd_values) > 1 else 0
                    }

                    return summary
            except Exception as e:
                print(f"Error reading summary stats: {e}")
                return None
        except Exception as e:
            print(f"Error reading summary stats: {e}")
            return None

    def plot_metrics(self, save_plots=True):
        """Generate plots for MW fairness metrics"""
        try:
            import matplotlib.pyplot as plt
            import pandas as pd

            # Read metrics
            df = pd.read_csv(self.metrics_file)
            df_numeric = df[df['round'] != 'final'].copy()
            df_numeric['round'] = pd.to_numeric(df_numeric['round'])

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f'{self.args.model} - MW Fairness Metrics', fontsize=16)

            # Plot 1: Accuracy over rounds
            axes[0, 0].plot(df_numeric['round'], df_numeric['mean_acc'], 'b-', linewidth=2)
            axes[0, 0].set_xlabel('Round')
            axes[0, 0].set_ylabel('Mean Accuracy (%)')
            axes[0, 0].set_title('Model Accuracy')
            axes[0, 0].grid(True, alpha=0.3)

            # Plot 2: TPRD over rounds
            axes[0, 1].plot(df_numeric['round'], df_numeric['tprd'], 'r-', linewidth=2)
            axes[0, 1].set_xlabel('Round')
            axes[0, 1].set_ylabel('TPRD')
            axes[0, 1].set_title('TPR Discrepancy (TPRD)')
            axes[0, 1].grid(True, alpha=0.3)

            # Plot 3: Group TPRs
            colors = ['green', 'orange', 'purple', 'brown', 'pink']
            for g in range(5):
                col = f'group_{g}_mean_tpr'
                if col in df_numeric.columns and df_numeric[col].notna().any():
                    axes[1, 0].plot(df_numeric['round'], df_numeric[col],
                                    label=f'Group {g}', color=colors[g], linewidth=2)
            axes[1, 0].set_xlabel('Round')
            axes[1, 0].set_ylabel('Mean TPR (%)')
            axes[1, 0].set_title('Group Mean TPRs')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

            # Plot 4: WTPR and BTPR
            axes[1, 1].plot(df_numeric['round'], df_numeric['wtpr'], 'g--',
                            label='Worst TPR', linewidth=2)
            axes[1, 1].plot(df_numeric['round'], df_numeric['btpr'], 'b--',
                            label='Best TPR', linewidth=2)
            axes[1, 1].fill_between(df_numeric['round'],
                                    df_numeric['wtpr'],
                                    df_numeric['btpr'],
                                    alpha=0.3, color='gray')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('TPR (%)')
            axes[1, 1].set_title('TPR Range (Worst to Best)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()

            if save_plots:
                plot_file = os.path.join(self.save_dir, f"{self.base_name}_plots.png")
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                print(f"Plots saved to {plot_file}")

            plt.show()

        except ImportError:
            print("Matplotlib or pandas not available for plotting")
        except Exception as e:
            print(f"Error generating plots: {e}")
