import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../core/model/")))
import data
import torch
import training
import lpips
import copy
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

class MetricExtractor:
    def __init__(self, model):
        self.model = model
        self.loss_functions = {"LPIPS_notune":lpips.LPIPS(net='alex',lpips=False).to(self.model.device),
                               "LPIPS_tuned":lpips.LPIPS(net='alex',lpips=True).to(self.model.device),
                               "SSIM":SSIM().to(self.model.device),
                               "L2":torch.nn.MSELoss()}

    def __call__(self, image_path):
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
        activations = self.model.get_activations(self.image, layer)
        activations = activations.flatten(start_dim=1)
        return activations

    def calculate_metrics(self, activations, layer):
        metrics = {}
        if layer in self.model.attention_layers:
            metrics["attention_"+layer] = torch.sum(torch.abs(activations)).cpu()
        else:
            metrics["L1_"+layer] = torch.sum(torch.abs(activations)).cpu()
            metrics["Gini_"+layer] = self.calculate_gini(activations)
            metrics["kurtosis_"+layer] = (((activations - activations.mean())**4).mean() / (activations.std()**4)).cpu()
        return metrics

    def calculate_gini(self, activations):
        activations = activations - torch.min(activations)
        n = activations.shape[1]
        activations = torch.sort(activations.cpu(),dim=1)[0].to(self.model.device)
        activations = torch.cumsum(activations,dim=1)
        gini = ((n + 1 - 2 * activations.sum(dim=1) / activations[:,-1]).cpu() / n)
        return gini.cpu()

    def get_reco_error(self, loss_name):
        prediction, _, _ = self.model(self.image)
        loss = self.loss_functions[loss_name](prediction, self.image).flatten()
        return {loss_name: loss.detach().cpu()}

    def get_SAM_delta(self, loss_name):
        model_copy = copy.deepcopy(self.model)#keep copy to revert to former version at the end
        opt = training.SAM(self.model.parameters(), torch.optim.Adam)
        prediction, _, _ = self.model(self.image)
        loss = self.loss_functions[loss_name](prediction, self.image)
        loss.backward()
        opt.first_step(zero_grad=True)
        prediction_max, _, _ = self.model(self.image)
        loss_max = self.loss_functions[loss_name](prediction_max, self.image)
        self.model = model_copy#revert to former version
        return {f"SAM_{loss_name}": (loss_max - loss).detach().cpu()}