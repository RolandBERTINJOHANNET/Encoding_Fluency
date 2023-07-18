import torch
import os
import functools
import warnings
#change warnings format
def custom_formatwarning(msg, *args, **kwargs):
    # ignore everything except the message
    return "\n!!!!!\n"+str(msg) + '\n\n'
warnings.formatwarning = custom_formatwarning
# Store the original warnings.warn function
original_warn = warnings.warn

def custom_warn(message, category=None, stacklevel=1, source=None):
    return original_warn(message, category, stacklevel=2, source=source)
warnings.warn = custom_warn

from torchvision.io import read_image
from torchvision.transforms import Resize
from torch.utils.data import Dataset
from PIL import Image

class OptionalSplitDataset(Dataset):

    warned_size = False
    warned_channels = False
    """
    A custom **PyTorch Dataset** for loading image data.

    This class is used to create a dataset object that can be used with a PyTorch DataLoader for efficient data loading.

    **Attributes**:
        device (torch.device): The device to load the images onto.
        file_paths (list): A list of paths to the image files.

    **Args**:
        root_dir (str): The root directory containing the image files.
        split (str): The dataset split ('train': keep 80% or 'test' (keep the other 20%) or 'none' (keep 100%)).
        device (torch.device): The device to load the images onto.
        
    .. code-block:: python
    
        # Example 1: Creating a dataset for training images on the CPU
        train_dataset = OptionalSplitDataset('/path/to/images', split='train', device=torch.device('cpu'))
    
        # Example 2: Creating a dataset for test images on the GPU
        test_dataset = OptionalSplitDataset('/path/to/images', split='test', device=torch.device('cuda'))
    
        # Example 3: Accessing an image from the dataset
        image = train_dataset[0]
    
        # Example 4: Getting the total number of images in the dataset
        num_images = len(train_dataset)
    
        # Example 5: Creating a dataset without splitting the data
        full_dataset = OptionalSplitDataset('/path/to/images', split='none', device=torch.device('cpu'))
    """
    def __init__(self, root_dir, split="train", device=None):
        """
        The constructor for OptionalSplitDataset class.

        This method initializes the dataset object, it sets the device where the images will be loaded, 
        gets all the image file paths from the root directory and sets the split for the dataset.

        **Args**:
            root_dir (str): The root directory containing the image files.
            split (str): The dataset split ('train' or 'test').
            device (torch.device): The device to load the images onto.
        """
        self.device = device
        self.file_paths = self.get_all_image_paths(root_dir)
        
        if self.device is not torch.device("cpu") and not torch.cuda.is_available():
            warnings.warn("The specified device is not available. Defaulting to CPU.")
            self.device = torch.device('cpu')
        
        if split not in ["train", "test", "none"]:
            raise ValueError("Invalid split. Must be either 'train', 'test' or 'none'.")
        
        split_index = int(len(self.file_paths) * 0.8)
        
        if split == "train":
            self.file_paths = self.file_paths[:split_index]
        elif split == "test":
            self.file_paths = self.file_paths[split_index:]
        elif split=="none":
            pass
        
    def get_all_image_paths(self, root_dir):
        """
        Recursively get all image paths from the root directory.

        This method is used to get all the image file paths from the root directory.

        **Args**:
            root_dir (str): The root directory containing the image files.

        **Returns**:
            list: A list of paths to the image files.
        """
        image_paths = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(root, file))
        return image_paths

    def __len__(self):
        """
        Get the total number of images in the dataset.

        This method is used to get the total number of images in the dataset.

        **Returns**:
            int: The total number of images.
        """
        return len(self.file_paths)
    
    @staticmethod
    def process_image(image_path, device):
        """
        Process an image.

        This method reads an image from a file, checks its size, resizes it to 224x224,
        adds a batch dimension, and moves it to the specified device.

        Args:
            image_path (str): The path to the image file.
            device (torch.device): The device to move the image to.

        Returns:
            torch.Tensor: The processed image tensor.
        """
        # Read the image using torchvision's read_image function
        image = read_image(image_path).float() / 255.0  # Normalize to [0, 1]

        # Check the number of channels in the image
        if image.shape[0] > 3:
            if not OptionalSplitDataset.warned_channels:
                warnings.warn("Image has more than 3 channels! Only the first 3 (RGB) channels will be used.", RuntimeWarning)
                OptionalSplitDataset.warned_channels = True
            image = image[:3, :, :]  # Keep only the first 3 channels

        elif image.shape[0] < 3:
            raise ValueError("Image has less than 3 channels! Please provide an RGB image.")

        # Check if the image is not 224x224 and issue a warning
        if image.shape[1] != 224 or image.shape[2] != 224:
            if not OptionalSplitDataset.warned_size:
                warnings.warn("Images in the input are not 224*224! Images are being"
                              " resized to 224x224 on the fly, which might make training somewhat slower", RuntimeWarning)
                OptionalSplitDataset.warned_size = True
            image = Resize((224, 224))(image)# Resize the image to 224x224
        
        # move the image to the specified device
        image = image.to(device)
        
        return image
    
    def __getitem__(self, index):
        """
        Get an image by index.

        This method is used to get an image and its label by index.

        Args:
            index (int): The index of the image.

        Returns:
            torch.Tensor: The image tensor.
        """
        file_path = self.file_paths[index]
        # Check if the file is a valid image file
        try:
            Image.open(file_path)
        except IOError:
            raise ValueError(f"The file {file_path} is not a valid image file.")
        
        return self.process_image(file_path, self.device)
