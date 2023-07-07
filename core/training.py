import matplotlib.pyplot as plt
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

            plot_inNout(dataset, model, epoch,model.name)

    return reconstruction_loss_per_batch, kl_divergence_per_batch, attention_per_batch


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