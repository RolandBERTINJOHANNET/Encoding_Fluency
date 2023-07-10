"""
This module defines a **Variational Autoencoder (VAE)** model with a unique architecture designed for image data. 
The model allows for the imposition of a *sparsity constraint* on certain layers, which can be useful for learning 
more compact or interpretable representations. The model is composed of an encoder, which uses a VGG19 architecture 
and includes a pixel shuffling operation, and a decoder, which gradually upscales the feature maps back to the 
original image size. The model also includes a method for accessing the activations of any layer.

.. code-block:: python

    Example usage:
        model = Model('model_name', 'cuda', [1, 2, 3], [1, 2], 1., 0.001)
        x = torch.randn(1, 3, 224, 224)
        output, kl, att = model(x)

"""

import torch.nn as nn
import torch
from core.model import basic_modules

class PixelShuffler(nn.Module):
    """
    A module that performs **pixel shuffling**, which is a way of upscaling the feature maps without 
    introducing any new parameters. The upscale factor determines the factor by which the spatial 
    dimensions are increased.

    :param upscale_factor: The factor by which to upscale the feature maps.
    :type upscale_factor: int
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return nn.functional.pixel_shuffle(x, self.upscale_factor)

class Model(nn.Module):
    """
    A **Variational Autoencoder (VAE)** model with a unique architecture designed for image data. The model allows 
    for the imposition of a *sparsity constraint* on certain layers.

    :param model_name: The name of the model.
    :type model_name: str
    :param device: The device to run the model on ('cpu' or 'cuda').
    :type device: str
    :param layer_sparsity_cstraint: A list of indices of layers to apply the sparsity constraint to.
    :type layer_sparsity_cstraint: list, optional
    :param attention: A list of attention values.
    :type attention: list, optional
    :param sparsity_coeff: The coefficient for the sparsity constraint.
    :type sparsity_coeff: float, optional
    :param sparsity_param: The parameter for the sparsity constraint.
    :type sparsity_param: float, optional
    """
    def __init__(self, model_name, device, layer_sparsity_cstraint=[], attention=[], sparsity_coeff=1., sparsity_param=0.001):
        super().__init__()
        self.latent_dim = 512
        self.name = model_name
        self.device = torch.device(str(device))
        self.encoder = self.build_encoder(layer_sparsity_cstraint, attention, sparsity_param, sparsity_coeff)
        self.decoder = self.build_decoder()
        self.layers = self.encoder[0].layers
        self.attention_layers = self.encoder[0].attention_layers
        self.all_layers = self.layers + [self.encoder[1].name + ("_mu"), self.encoder[1].name + ("_std"), self.encoder[2].name]

    def build_encoder(self, layer_sparsity_cstraint, attention, sparsity_param, sparsity_coeff):
        modules = []
        modules.append(basic_modules.VGG19_Features(layer_sparsity_cstraint, attention, sparsity_param, sparsity_coeff))
        modules.append(basic_modules.MeanStdFeatureMaps(self.latent_dim, self.latent_dim))
        modules.append(basic_modules.Reparametrization(self.device))
        encoder = nn.Sequential(*modules)
        for layer_idx in [index-15 for index in layer_sparsity_cstraint if index>=16]:
            if layer_idx>2:
                raise ValueError("you provided an out-of-bounds index for the sparsity constraint location !")
            encoder[layer_idx].add_constraint(sparsity_param,sparsity_coeff)
        return encoder

    def build_decoder(self):
        modules = []
        upscale_factor = 2
        in_channels = self.latent_dim
        nb_filters = [self.latent_dim, 512, 256, 128, 64]
        for i in range(len(nb_filters) * 2):
            if i % 2 == 0:
                out_chan = int(in_channels * upscale_factor)
                modules.append(nn.Sequential(nn.Conv2d(in_channels, out_chan, 3, 1, padding=1), nn.BatchNorm2d(out_chan), PixelShuffler(upscale_factor=upscale_factor), nn.LeakyReLU()))
                in_channels = int(out_chan / (upscale_factor ** 2))
            else:
                modules.append(basic_modules.ResBlock(in_channels))
        out_chan = int(in_channels * upscale_factor)
        modules.append(nn.Sequential(nn.Conv2d(in_channels, 3, 3, 1, 1, padding_mode="reflect"), nn.Sigmoid()))
        return nn.Sequential(*modules)

    def forward(self, x):
        """
        Passes the input through the model.

        :param x: The input tensor.
        :type x: torch.Tensor
        :returns: A tuple containing the output tensor, the KL divergence, and the attention values.
        :rtype: tuple
        """
        kl_terms = []
        x, kl, att = self.encoder[0](x)
        kl_terms.append(kl)
        x, kl = self.encoder[1](x)
        kl_terms.append(kl)
        x, kl = self.encoder[2](x)
        kl_terms.append(kl)
        kl = sum(kl_terms)
        for module in self.decoder:
            x = module(x)
        return x, kl, att

    def get_activations(self, x, layer_name):
        """
        Returns the activations of a specific layer.

        :param x: The input tensor.
        :type x: torch.Tensor
        :param layer_name: The name of the layer.
        :type layer_name: str
        :returns: The activations of the layer.
        :rtype: torch.Tensor
        """
        activations = self.encoder[0].get_activations(x, layer_name)
        if activations is not None:
            return activations
        x = self.encoder[0](x)[0]
        if "MeanStdFeatureMaps" in layer_name:
            return self.encoder[1].get_activations(x, "mu" if "mu" in layer_name else "sigma")
        x = self.encoder[1](x)[0]
        if "Reparametrization" in layer_name:
            return self.encoder[2].get_activations(x)
        raise ValueError(f"Invalid layer name: {layer_name}")

    def encode(self, x, sample=False):
        """
        Encodes the input into the latent space.

        :param x: The input tensor.
        :type x: torch.Tensor
        :param sample: Whether to sample from the latent space.
        :type sample: bool, optional
        :returns: The encoded tensor.
        :rtype: torch.Tensor
        """
        for layer in self.encoder[:-1]:
            x, _ = layer(x)
        return self.encoder[-1](x)[0] if sample else x[:, :int(x.shape[1] / 2)]

    def decode(self, x):
        """
        Decodes the input from the latent space.

        :param x: The input tensor.
        :type x: torch.Tensor
        :returns: The decoded tensor.
        :rtype: torch.Tensor
        """
        return self.decoder(x)