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

    This function takes a MetricExtractor instance, a dataset, a file path, and an optional number of images to plot.
    It then plots activation histograms for each layer of the model for each image, and saves the plot to a specified file.

    **Parameters:**

    - **metric_extractor** (*MetricExtractor instance*): The MetricExtractor instance. This should be properly initialized and should have a model attribute.

    - **dataset** (*Dataset instance*): The dataset containing the images. This should have a file_paths attribute which is an iterable of image file paths.

    - **file_path** (*str*): The path where the plot should be saved. This should be a valid directory where the user has write permissions.

    - **num_images** (*int, optional*): The number of images to plot. Default is 5. This should be a positive integer.

    **Raises:**

    - **ValueError**: If any of the inputs are not as expected.

    **Example usage:**

    .. code-block:: python
    
        from data import OptionalSplitDataset
        from core.metrics import MetricExtractor

        # Initialize a MetricExtractor instance
        metric_extractor = MetricExtractor(model, device)

        # Initialize a dataset
        dataset = ImageFolder('path_to_dataset')

        # Specify the file path
        file_path = 'path_to_save_plot'

        # Call the function
        plot_activation_histograms(metric_extractor, dataset, file_path, num_images=10)

    """
    if not os.path.isdir(file_path):
        raise ValueError("file_path must be a valid directory.")
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

    This function takes a data loader, a model, an epoch or output path, and a model name. It then plots the original images and their reconstructions side by side, and saves the plot to a specified file.

    **Parameters:**

    - **data** (*DataLoader instance*): The DataLoader instance providing batches of images.

    - **model** (*nn.Module instance*): The PyTorch model to use for reconstruction. This should be a model that can take a batch of images and return their reconstructions.

    - **epoch** (*int or str*): The current epoch (for saving purposes), or if during metrics extraction, the desired output path.

    - **model_name** (*str*): The name of the model. This is used for saving purposes.

    **Raises:**

    - **ValueError**: If any of the inputs are not as expected.

    **Example usage:**

    .. code-block:: python

        from torch.utils.data import DataLoader
        from torchvision.transforms import ToTensor
        from core.model.model import Model
        from core.data import OptionalSplitDataset

        # Initialize a DataLoader instance
        dataset = OptionalSplitDataset(root_dir='path_to_dataset', split='none', device='cuda')
        data = DataLoader(dataset, batch_size=64)

        # Initialize a model
        model = Model(model_name='my_model', device='cuda')

        # Specify the epoch and model name
        epoch = 10
        model_name = 'my_model'

        # Call the function
        plot_inNout(data, model, epoch, model_name)

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
