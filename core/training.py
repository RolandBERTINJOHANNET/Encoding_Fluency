import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/model/")))
import plotting
import torch
import tqdm

def train_the_model(model, dataset, loss, opt, nb_epochs):
    """
    Train a model for a specified number of epochs.

    This function takes a model, a dataset, a loss function, an optimizer, and a number of epochs, and trains the model on the dataset for the specified number of epochs. It returns the reconstruction loss and KL divergence for each batch.

    **Parameters:**

    - **model** (*nn.Module instance*): The PyTorch model to train.

    - **dataset** (*DataLoader instance*): The DataLoader instance providing batches of images.

    - **loss** (*nn.Module instance*): The PyTorch loss function to use.

    - **opt** (*torch.optim.Optimizer instance*): The PyTorch optimizer to use.

    - **nb_epochs** (*int*): The number of epochs to train for.

    - **save** (*int, optional*): The frequency at which to save the model. Default is 10.

    **Returns:**

    - **tuple**: A tuple of three lists: the reconstruction loss per batch, the KL divergence per batch, and the attention loss per batch.

    **Raises:**

    - **ValueError**: If the dataset is empty.

    **Example usage:**

    .. code-block:: python

        from torch.utils.data import DataLoader
        from torchvision.transforms import ToTensor
        from torch import nn, optim
        from core.model.models import Model
        from core.data import OptionalSplitDataset

        # Initialize a DataLoader instance
        dataset = OptionalSplitDataset(root_dir='path_to_dataset', split='none', device='cuda')
        data = DataLoader(dataset, batch_size=64)

        # Initialize a model
        model = Model(model_name='my_model', device='cuda')

        # Specify the loss function and optimizer
        loss = nn.MSELoss()
        opt = optim.Adam(model.parameters())

        # Specify the number of epochs and save frequency
        nb_epochs = 100
        save = 10

        # Call the function
        reconstruction_loss_per_batch, kl_divergence_per_batch, attention_per_batch = train_the_model(model, data, loss, opt, nb_epochs, save)

    """
    if not dataset:
        raise ValueError("Dataset is empty.")

    reconstruction_loss_per_batch = []
    kl_divergence_per_batch = []
    attention_per_batch = []

    for epoch in range(nb_epochs):
        batch_iter = tqdm.tqdm(dataset, unit="batch", total=len(dataset))
        for X_b in batch_iter:
            kl_divergence = None  # reinit kl loss
            pred, kl_divergence,attention_L1_norm = model(X_b)
            reconstruction_loss = loss(pred, X_b).mean()

            reconstruction_loss_per_batch.append(reconstruction_loss.item())
            kl_divergence_per_batch.append(float(kl_divergence))
            attention_per_batch.append(float(attention_L1_norm))

            opt.zero_grad()

            (reconstruction_loss + kl_divergence + attention_L1_norm).backward()
            opt.step()
            batch_iter.set_description(f"reco loss: {reconstruction_loss.item()}, kl loss: {float(kl_divergence)}, attention loss: {float(attention_L1_norm)}")

            plotting.plot_inNout(dataset, model, epoch,model.name)

    return reconstruction_loss_per_batch, kl_divergence_per_batch, attention_per_batch




#this is taken straight from https://github.com/davda54/sam
#in this project we only use this to perform the first_step when extracting the sharpness metric
class SAM(torch.optim.Optimizer):
    """
    This class implements the Sharpness-aware-Minimization optimizer, taken from `git <https://github.com/davda54/sam>`_.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    print("no grad !")
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups