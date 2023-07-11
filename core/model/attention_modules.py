import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    """
    This class is a basic convolutional layer, taken from `git <https://github.com/Jongchan/attention-module/tree/master>`_.. It includes options for batch normalization and ReLU activation.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    """
    This class is a simple layer that flattens its input, taken from `git <https://github.com/Jongchan/attention-module/tree/master>`_..
    """
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    """
    This class implements a channel gate, taken from `git <https://github.com/Jongchan/attention-module/tree/master>`_.. It includes options for different types of pooling.
    """
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        l1 = channel_att_sum.abs().mean()#when I apply after the sigmoid it gets stuck and doesn't go down allt he way to 0
        
        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        
        return x * scale,l1
    def get_att_map(self,x):
        with torch.no_grad():
            channel_att_sum = None
            for pool_type in self.pool_types:
                if pool_type=='avg':
                    avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                    channel_att_raw = self.mlp( avg_pool )
                elif pool_type=='max':
                    max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                    channel_att_raw = self.mlp( max_pool )
                elif pool_type=='lp':
                    lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                    channel_att_raw = self.mlp( lp_pool )
                elif pool_type=='lse':
                    # LSE pool only
                    lse_pool = logsumexp_2d(x)
                    channel_att_raw = self.mlp( lse_pool )
    
                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw
            return channel_att_sum

def logsumexp_2d(tensor):
    """
    This function calculates the log-sum-exp of a 2D tensor, taken from `git <https://github.com/Jongchan/attention-module/tree/master>`_.
    """
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    """
    This class implements a channel pooling layer, taken from `git <https://github.com/Jongchan/attention-module/tree/master>`_.. It returns the concatenation of the max and mean of its input.
    """
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

#the main module --
class Constrained_CBAM(nn.Module):
    """
    This class implements a modified version of the Convolutional Block Attention Module (CBAM) with a L1 constraint. 
    It only includes depth-wise attention, also known as channel attention.

    **Attributes**:
        gate_channels (int): The number of output channels in the attention module.
        constraint_param (float): The parameter for the L1 constraint.
        scaling_param (float): The scaling parameter for the attention module.
        index (int): The index of the attention module in the model.
        reduction_ratio (int, optional): The ratio for the channel reduction in the attention module. Default is 16.
        pool_types (list of str, optional): The types of pooling to use in the attention module. Default is ['avg', 'max'].
        no_spatial (bool, optional): Whether to exclude spatial attention. Default is False.

    **Example usage**:

    .. code-block:: python

        from model import Constrained_CBAM

        # Initialize the Constrained_CBAM module
        cbam = Constrained_CBAM(gate_channels=64, constraint_param=0.1, scaling_param=0.5, index=1)

        # Use the Constrained_CBAM module on an input
        output, l1 = cbam(input)

    """
    def __init__(self, gate_channels, constraint_param, scaling_param, index, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        """
        Initialize the Constrained_CBAM module.

        **Parameters**:
            gate_channels (int): The number of output channels in the attention module.
            constraint_param (float): The parameter for the L1 constraint.
            scaling_param (float): The scaling parameter for the attention module.
            index (int): The index of the attention module in the model.
            reduction_ratio (int, optional): The ratio for the channel reduction in the attention module. Default is 16.
            pool_types (list of str, optional): The types of pooling to use in the attention module. Default is ['avg', 'max'].
            no_spatial (bool, optional): Whether to exclude spatial attention. Default is False.
        """
        super().__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.name = "attention"+str(index)+"_Sigmoid"
        
    def forward(self, x):
        """
        Forward pass of the Constrained_CBAM module.

        **Parameters**:
            x (torch.Tensor): The input tensor.

        **Returns**:
            tuple: The output tensor and the L1 norm.
        """
        l1=0.
        x_out,l1 = self.ChannelGate(x)
        return x_out,l1
    
    def get_activations(self,x):
        """
        Get the activation map of the Constrained_CBAM module.

        **Parameters**:
            x (torch.Tensor): The input tensor.

        **Returns**:
            torch.Tensor: The activation map.
        """
        return self.ChannelGate.get_att_map(x)