import sys
sys.path.append("../../core/","../../core/model/")
import data
import torch
import SAM_opt
import lpips
from torchmetrics import StructuralSimilarityIndexMeasure as SSIM

class MetricExtractor:
    def __init__(self, model, image_path):
        self.model = model
        self.image = data.CustomDataset.process_image(image_path, torch.device("cpu"))
        self.loss_functions = {"LPIPS_notune":lpips.LPIPS(net='alex',lpips=False),
                               "LPIPS_tuned":lpips.LPIPS(net='alex',lpips=True),
                               "SSIM":SSIM(),
                               "L2":torch.nn.MSELoss()}

    def __call__(self, image_path):
        self.image = data.CustomDataset.process_image(image_path, torch.device("cpu"))
        metrics = {}
        for layer in self.layers:
            metrics.update(self.get_metrics(layer))
        for loss_name in self.loss_functions.keys():
            metrics.update(self.get_reco_error(loss_name))
            metrics.update(self.get_SAM_delta(loss_name))
        return metrics

    def get_activations(self, layer):
        activations = self.model.get_activations(self.image, layer)
        activations = activations.flatten(start_dim=1)
        return activations

    def calculate_metrics(self, activations, layer):
        metrics = {}
        metrics["L1_"+layer] = torch.sum(torch.abs(activations)).cpu()
        metrics["Gini_"+layer] = self.calculate_gini(activations)
        metrics["kurtosis_"+layer] = torch.kurtosis(activations).cpu()
        if layer in self.model.attention_layers:
            metrics["attention_"+layer] = torch.sum(torch.abs(activations)).cpu()
        return metrics

    def calculate_gini(self, activations):
        activations = activations - torch.min(activations)
        n = activations.shape[1]
        activations = torch.sort(activations.cpu())[0].to(torch.device("cuda:0"))
        activations = torch.cumsum(activations)
        gini = (n + 1 - 2 * torch.sum(activations) / activations[-1]) / n
        return gini.cpu()

    def get_reco_error(self, loss_name):
        prediction, _ = self.model(self.image)
        loss = self.loss_functions[loss_name](prediction, self.image)
        return {loss_name: loss.detach().cpu()}

    def get_SAM_delta(self, loss_name):
        original_state = [p.clone() for p in self.model.parameters()]
        opt = SAM_opt.SAM(self.model.parameters(), torch.optim.Adam)
        prediction, _ = self.model(self.image)
        loss = self.loss_functions[loss_name](prediction, self.image)
        loss.backward()
        opt.first_step(zero_grad=True)
        prediction_max, _ = self.model(self.image)
        loss_max = self.loss_functions[loss_name](prediction_max, self.image)
        for p, original in zip(self.model.parameters(), original_state):
            p.data = original
        return {"SAM": (loss_max - loss).detach().cpu()}
