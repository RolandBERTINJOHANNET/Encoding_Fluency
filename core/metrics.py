"""
This module contains the `MetricExtractor` class, a utility for extracting various metrics from a given model. 
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/model/")))
import data
import torch
import training
import lpips
import copy
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class MetricExtractor:
    """
    A class for extracting various metrics from a given model.

    This class is initialized with a model and sets up several loss functions. It is primarily used to calculate and return a dictionary of metrics for a given image. These metrics include layer-wise activations, L1 norms, Gini coefficients, kurtosis, and reconstruction errors based on different loss functions. The class also calculates the Stochastic Weight Averaging Gaussian (SWAG) delta for each loss function.

    **Attributes**:
        model (nn.Module): The PyTorch model from which to extract metrics.
        loss_functions (dict): A dictionary of loss functions.

    **Example usage**:

    .. code-block:: python

        from core.model.model import Model

        # Initialize a model
        model = Model(model_name='my_model', device='cuda')

        # Initialize the MetricExtractor
        metric_extractor = MetricExtractor(model)

        # Use the MetricExtractor on an image
        metrics = metric_extractor('path_to_image.jpg')

    """
    def __init__(self, model):
        """
        Initialize the MetricExtractor with a model.

        **Parameters**:
            model (nn.Module): The PyTorch model from which to extract metrics.
        """
        self.model = model
        self.loss_functions = {"LPIPS_notune":lpips.LPIPS(net='alex',lpips=False).to(self.model.device),
                               "LPIPS_tuned":lpips.LPIPS(net='alex',lpips=True).to(self.model.device),
                               "SSIM":SSIM().to(self.model.device),
                               "L2":torch.nn.MSELoss()}

    def __call__(self, image_path):
        """
        Calculate and return a dictionary of metrics for a given image.

        **Parameters**:
            image_path (str): The path to the image.

        **Returns**:
            dict: A dictionary of metrics.
        """
        self.image = data.OptionalSplitDataset.process_image(image_path, self.model.device)[None,:]
        metrics = {}
        for layer in self.model.layers + self.model.attention_layers:
            with torch.no_grad():
                activations = self.get_activations(layer)
                metrics.update(self.calculate_metrics(activations, layer))
        for loss_name in self.loss_functions.keys():
            with torch.no_grad():
                metrics.update(self.get_reco_error(loss_name))
            metrics.update(self.get_SAM_delta(loss_name))
        return metrics


    def get_activations(self, layer):
        """
        Get the activations of a given layer for the current image.

        **Parameters**:
            layer (str): The name of the layer.

        **Returns**:
            torch.Tensor: The activations of the layer.
        """
        activations = self.model.get_activations(self.image, layer)
        activations = activations.flatten(start_dim=1)
        return activations

    def calculate_metrics(self, activations, layer):
        """
        Calculate and return a dictionary of metrics for a given layer's activations.
    
        The metrics calculated depend on whether the layer is an attention layer or not. For attention layers, the sum of absolute activations is calculated. For other layers, the L1 norm, Gini coefficient, and kurtosis of the activations are calculated.
    
        **Parameters**:
            activations (torch.Tensor): The activations of the layer.
            layer (str): The name of the layer.
    
        **Returns**:
            dict: A dictionary of metrics for the layer.
        """
        metrics = {}
        if layer in self.model.attention_layers:
            metrics["attention_"+layer] = torch.sum(torch.abs(activations)).cpu()
        else:
            metrics["L1_"+layer] = torch.sum(torch.abs(activations)).cpu()
            metrics["Gini_"+layer] = self.calculate_gini(activations)
            metrics["kurtosis_"+layer] = (((activations - activations.mean())**4).mean() / (activations.std()**4)).cpu()
        return metrics

    def calculate_gini(self, activations):
        """
        Calculate the reconstruction error for the current image using a specified loss function.
    
        **Parameters**:
            loss_name (str): The name of the loss function to use.
    
        **Returns**:
            dict: A dictionary with the loss name as the key and the reconstruction error as the value.
        """
        activations = activations - torch.min(activations)
        n = activations.shape[1]
        activations = torch.sort(activations.cpu(),dim=1)[0].to(self.model.device)
        activations = torch.cumsum(activations,dim=1)
        gini = ((n + 1 - 2 * activations.sum(dim=1) / activations[:,-1]).cpu() / n)
        return gini.cpu()

    def get_reco_error(self, loss_name):
        """
        Calculate the reconstruction error for the current image using a specified loss function.
    
        **Parameters**:
            loss_name (str): The name of the loss function to use.
    
        **Returns**:
            dict: A dictionary with the loss name as the key and the reconstruction error as the value.
        """
        prediction, _, _ = self.model(self.image)
        loss = self.loss_functions[loss_name](prediction, self.image).flatten()
        return {loss_name: loss.detach().cpu()}

    def get_SAM_delta(self, loss_name):
        """
        Calculate the Sharpness-aware-Minimization -inspired delta for a specified loss function.
    
        This metric measures how much a small change in model parameters can collapse its ability to reconstruct an image'
    
        **Parameters**:
            loss_name (str): The name of the loss function to use.
    
        **Returns**:
            dict: A dictionary with the loss name as the key and the sam delta as the value.
        """
        model_copy = copy.deepcopy(self.model)#keep copy to revert to former version at the end
        opt = training.SAM(self.model.parameters(), torch.optim.Adam)
        prediction, _, _ = self.model(self.image)
        loss = self.loss_functions[loss_name](prediction, self.image)
        loss.backward()
        opt.first_step(zero_grad=True)
        prediction_max, _, _ = self.model(self.image)
        loss_max = self.loss_functions[loss_name](prediction_max, self.image)
        self.model = model_copy#revert to former version
        return {f"SAM_{loss_name}": (loss_max - loss).detach().cpu().flatten()}