# src/utils/group_assignment.py

import torch
import numpy as np


def assign_groups_by_features(data, num_groups, method='kmeans'):
    """
    Assign data points to groups based on their features.

    Args:
        data: Tensor of data features
        num_groups: Number of groups to assign
        method: Method to use for assignment ('kmeans', 'threshold', etc.)

    Returns:
        List of group assignments for each data point
    """
    if method == 'kmeans':
        # Simple k-means clustering implementation
        # In a real implementation, you'd use a proper clustering library
        centers = data[np.random.choice(len(data), num_groups, replace=False)]

        assignments = []
        for x in data:
            dists = [np.linalg.norm(x.cpu().numpy() - c.cpu().numpy()) for c in centers]
            assignments.append(np.argmin(dists))

        return assignments

    elif method == 'threshold':
        # Simple threshold-based assignment
        # For example, assign based on the first feature being above/below a threshold
        if num_groups == 2:
            threshold = data[:, 0].median()
            return [0 if x[0] < threshold else 1 for x in data]
        else:
            # Divide the first feature into num_groups quantiles
            thresholds = np.percentile(data[:, 0].cpu().numpy(),
                                       np.linspace(0, 100, num_groups + 1)[1:-1])

            assignments = []
            for x in data:
                val = x[0].item()
                group = 0
                for i, t in enumerate(thresholds):
                    if val >= t:
                        group = i + 1
                assignments.append(group)

            return assignments

    else:
        raise ValueError(f"Unknown group assignment method: {method}")


def assign_groups_by_noise(noise_clients, parti_num, num_groups=2):
    """
    Assign clients to groups based on whether they have noise applied.

    Args:
        noise_clients: List of client indices with noise
        parti_num: Total number of participating clients
        num_groups: Number of groups to assign

    Returns:
        Dictionary mapping client indices to group assignments
    """
    # For binary case (noisy vs non-noisy)
    if num_groups == 2:
        return {i: 1 if i in noise_clients else 0 for i in range(parti_num)}

    # For more complex group assignments
    else:
        # Default assignment for non-noise clients
        assignments = {i: 0 for i in range(parti_num)}

        # Divide noise clients into num_groups-1 groups based on noise level
        noise_clients_sorted = sorted(noise_clients)
        groups_size = len(noise_clients) // (num_groups - 1)

        for i, client_idx in enumerate(noise_clients_sorted):
            group = (i // groups_size) + 1
            if group < num_groups:
                assignments[client_idx] = group
            else:
                assignments[client_idx] = num_groups - 1

        return assignments
