import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../core/model/")))
import plotting
import torch
import tqdm

def train_the_model(model, dataset, loss, opt, nb_epochs, save=10):
    """
    Train a model for a specified number of epochs.

    Args:
    model: The model to train.
    dataset: The dataset to train on.
    loss: The loss function to use.
    opt: The optimizer to use.
    nb_epochs: The number of epochs to train for.
    save: The frequency at which to save the model.

    Returns:
    A tuple of two lists: the reconstruction loss per batch and the KL divergence per batch.
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

class SAM(torch.optim.Optimizer):
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