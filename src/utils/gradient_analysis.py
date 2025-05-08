import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import torch
import torch.nn as nn
from tqdm import tqdm


def extract_gradients_from_fc_layer(model, dataloader, device, layer_name='fc'):
    """
    Extract gradients from a fully connected layer of a model.

    Args:
        model: Neural network model
        dataloader: Data loader for computing gradients
        device: Device to run computations on
        layer_name: Name of the fully connected layer to extract gradients from

    Returns:
        Flattened gradients from the specified layer
    """
    model.eval()
    model = model.to(device)

    # Find the target layer
    target_layer = None
    for name, module in model.named_modules():
        if layer_name in name and isinstance(module, nn.Linear):
            target_layer = module
            break

    if target_layer is None:
        # Try common FC layer names
        for name, module in model.named_modules():
            if (isinstance(module, nn.Linear) and
                    any(l in name for l in ['linear', 'fc', 'classifier'])):
                target_layer = module
                print(f"Found layer: {name}")
                break

    if target_layer is None:
        raise ValueError(f"Could not find a fully connected layer with name containing '{layer_name}'")

    all_gradients = []

    for data, target in tqdm(dataloader, desc="Extracting gradients"):
        data, target = data.to(device), target.to(device)
        model.zero_grad()

        # Add noise if applicable
        if hasattr(model, 'noise_variances') and model.noise_variances:
            # Apply appropriate noise if this is a specific client's model
            # For simplicity, we're not applying noise in this standalone function
            pass

        output = model(data)

        # Handle different model output formats
        if isinstance(output, tuple):
            output = output[0]  # Some models return (output, features)

        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        # Extract gradients
        if target_layer.weight.grad is not None:
            grad = target_layer.weight.grad.cpu().detach().numpy().flatten()
            all_gradients.append(grad)

    # Concatenate all batch gradients
    if all_gradients:
        return np.concatenate(all_gradients)
    else:
        return np.array([])


def analyze_gradient_correlation(model1, loader1, model2, loader2, device, save_path='gradient_correlation.png'):
    """
    Analyze and visualize the correlation between gradients of two models.

    Args:
        model1: First model
        loader1: Data loader for first model
        model2: Second model
        loader2: Data loader for second model
        device: Device to run computations on
        save_path: Path to save the visualization
    """
    # Extract gradients
    print("Extracting gradients from model 1...")
    gradients1 = extract_gradients_from_fc_layer(model1, loader1, device)

    print("Extracting gradients from model 2...")
    gradients2 = extract_gradients_from_fc_layer(model2, loader2, device)

    # Ensure we have gradients
    if len(gradients1) == 0 or len(gradients2) == 0:
        print("Failed to extract gradients!")
        return

    # Sample gradients if they're too large (for visualization purposes)
    max_samples = 5000
    if len(gradients1) > max_samples:
        indices = np.random.choice(len(gradients1), max_samples, replace=False)
        gradients1 = gradients1[indices]

    if len(gradients2) > max_samples:
        indices = np.random.choice(len(gradients2), max_samples, replace=False)
        gradients2 = gradients2[indices]

    # Calculate correlation
    correlation, _ = spearmanr(gradients1, gradients2)

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Add client 1 histogram
    ax1.hist(gradients1, bins=50, color='darkred', alpha=0.8)
    ax1.set_xlabel('Gradients')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Client 1 (pristine)')

    # Add client 2 histogram
    ax2.hist(gradients2, bins=50, color='teal', alpha=0.8)
    ax2.set_xlabel('Gradients')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Client 2 (noise)')

    fig.suptitle(f'Gradient distribution in a fully connected layer (Correlation={correlation:.2f})')
    plt.tight_layout()

    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    plt.close()

    return correlation


def plot_combined_gradient_correlation(models, loaders, client_names, noise_status, device,
                                       output_dir='visualizations'):
    """
    Create a publication-ready gradient correlation visualization for multiple clients.

    Args:
        models: List of models
        loaders: List of data loaders
        client_names: List of client names or identifiers
        noise_status: List of noise descriptions (e.g., 'pristine', 'noise=0.11')
        device: Device to run computations on
        output_dir: Directory to save the visualization
    """
    if len(models) < 2 or len(loaders) < 2:
        print("Need at least two models and loaders to compare")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Sample pairs of clients for comparison
    for i in range(0, len(models), 2):
        if i + 1 >= len(models):
            break

        model1, model2 = models[i], models[i + 1]
        loader1, loader2 = loaders[i], loaders[i + 1]
        name1, name2 = client_names[i], client_names[i + 1]
        status1, status2 = noise_status[i], noise_status[i + 1]

        # Extract gradients
        print(f"Extracting gradients from {name1}...")
        gradients1 = extract_gradients_from_fc_layer(model1, loader1, device)

        print(f"Extracting gradients from {name2}...")
        gradients2 = extract_gradients_from_fc_layer(model2, loader2, device)

        # Sample gradients if they're too large
        max_samples = 5000
        if len(gradients1) > max_samples:
            indices = np.random.choice(len(gradients1), max_samples, replace=False)
            gradients1 = gradients1[indices]

        if len(gradients2) > max_samples:
            indices = np.random.choice(len(gradients2), max_samples, replace=False)
            gradients2 = gradients2[indices]

        # Calculate correlation
        correlation, _ = spearmanr(gradients1, gradients2)

        # Create publication-style visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Client 1 histogram
        ax1.hist(gradients1, bins=30, color='darkred', alpha=0.8)
        ax1.set_xlabel('Gradients')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Client {name1} ({status1})')

        # Client 2 histogram
        ax2.hist(gradients2, bins=30, color='teal', alpha=0.8)
        ax2.set_xlabel('Gradients')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Client {name2} ({status2})')

        # Add correlation as a subfigure title
        save_path = os.path.join(output_dir, f'gradient_correlation_{name1}_{name2}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        # Create a second figure with correlation as title for paper-like visualization
        fig2 = plt.figure(figsize=(12, 5))

        # Add a title with correlation
        plt.figtext(0.5, 0.01, f'({"a" if correlation > 0 else "b"}) Correlation={correlation:.2f}',
                    ha='center', fontsize=12)

        # Two subplots
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)

        # Client 1 histogram
        ax1.hist(gradients1, bins=30, color='darkred', alpha=0.8)
        ax1.set_xlabel('Gradients')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'Client {name1} ({status1})')

        # Client 2 histogram
        ax2.hist(gradients2, bins=30, color='teal', alpha=0.8)
        ax2.set_xlabel('Gradients')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'Client {name2} ({status2})')

        plt.tight_layout()
        save_path2 = os.path.join(output_dir, f'paper_gradient_corr_{name1}_{name2}.png')
        plt.savefig(save_path2, dpi=300, bbox_inches='tight')

        print(f"Correlation between {name1} and {name2}: {correlation:.2f}")
        print(f"Visualizations saved to {save_path} and {save_path2}")
        plt.close('all')


# Integration with your existing code
def analyze_model_gradients(model, trainloaders, noise_clients, device, output_dir='visualizations'):
    """
    Analyze gradients from trained models with a focus on noise impact.

    Args:
        model: The federated learning model (should have nets_list attribute)
        trainloaders: List of training data loaders
        noise_clients: List of client indices with noise
        device: Device to run computations on
        output_dir: Directory to save visualizations
    """
    if not hasattr(model, 'nets_list'):
        print("Model doesn't have nets_list attribute")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Select some clients for analysis
    clients_to_analyze = []

    # Add some noisy clients
    for i in noise_clients[:min(2, len(noise_clients))]:
        clients_to_analyze.append(i)

    # Add some non-noisy clients
    non_noisy = [i for i in range(len(model.nets_list)) if i not in noise_clients]
    for i in non_noisy[:min(2, len(non_noisy))]:
        clients_to_analyze.append(i)

    # Prepare data for visualization
    models_list = [model.nets_list[i] for i in clients_to_analyze]
    loaders_list = [trainloaders[i] for i in clients_to_analyze]
    client_names = [str(i + 1) for i in clients_to_analyze]
    noise_status = ['noise=0.8' if i in noise_clients else 'pristine' for i in clients_to_analyze]

    # Run the analysis
    plot_combined_gradient_correlation(
        models_list, loaders_list, client_names, noise_status, device, output_dir
    )
    