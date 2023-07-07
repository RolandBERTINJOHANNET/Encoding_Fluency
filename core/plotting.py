import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/model/")))
import matplotlib.pyplot as plt
from data import OptionalSplitDataset
import torch
from tqdm import tqdm

def plot_activation_histograms(metric_extractor, dataset, file_path, num_images=5):
    """
    Plot activation histograms for a given number of images from a dataset.

    Args:
    metric_extractor: The MetricExtractor instance.
    dataset: The dataset containing the images.
    file_path: The path where the plot should be saved.
    num_images: The number of images to plot (default is 5).
    """
    # Create a figure for the plot
    num_layers = len(metric_extractor.model.layers + metric_extractor.model.attention_layers)
    fig, axs = plt.subplots(num_images, num_layers+1, figsize=(5*num_layers, 5*num_images))

    # Process each image in the dataset
    for i, image_path in tqdm(enumerate(dataset.file_paths[:num_images]), total=num_images, desc="plotting histograms",unit="image"):
        _ = metric_extractor(image_path)  # setup the metrics extractor

        # Plot the original image
        image = OptionalSplitDataset.process_image(image_path, metric_extractor.model.device)[None,:].cpu().numpy()
        axs[i, 0].imshow(image[0].transpose(1, 2, 0))
        axs[i, 0].axis('off')

        # Plot the activation histograms for each layer
        for j, layer in enumerate(metric_extractor.model.all_layers):
            activations = metric_extractor.get_activations(layer).cpu().numpy()
            axs[i, j+1].hist(activations.flatten(), bins=50)
            axs[i, j+1].set_title(layer)

    # Save the plot to a file
    plt.savefig(f"{file_path}/activation_histograms.png")
    plt.close(fig)


def plot_inNout(data, model, epoch, model_name):
    """
    Plot the original images and their reconstructions side by side.

    Args:
    data: The original images.
    model: The model to use for reconstruction.
    epoch: The current epoch (or, if during metrics extraction, the desired output path).
    model_name: The name of the model (for saving path purposes)
    """
    # Select a batch of data
    images = next(iter(data))

    # Pass the images through the model
    with torch.no_grad():
        try:
            reconstructions, _, _ = model(images)
        except Exception as e:
            print(f"Failed to generate reconstructions: {e}")
            return

    # Move the images and reconstructions to cpu and convert to numpy arrays
    images = images.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()

    # Create a figure for the plot
    fig, axs =plt.subplots(2, 3, figsize=(9, 6))


    # Plot the original images and the reconstructions
    for i in range(3):
        # Original images
        axs[0, i].imshow(images[i].transpose(1, 2, 0))
        axs[0, i].axis('off')

        # Reconstructions
        axs[1, i].imshow(reconstructions[i].transpose(1, 2, 0))
        axs[1, i].axis('off')

    # Save the plot to a file
    if isinstance(epoch,str):
        plt.savefig(f"{epoch}/in_out_plot.png")
        plt.close(fig)
    else:
        plt.savefig(f"{model_name}/in_out_plot_epoch_{epoch}.png")
        plt.close(fig)
