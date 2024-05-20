import os
import json
import torch
from unet import UNet, ConvPass

setup_dir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# load model parameters
with open(os.path.join(setup_dir, "config.json")) as f:
    net_config = json.load(f)

in_channels = net_config['in_channels']
output_shapes = net_config['output_shapes']
num_fmaps = net_config['num_fmaps']
fmap_inc_factor = net_config['fmap_inc_factor']
downsample_factors = eval(repr(net_config['downsample_factors']).replace('[', '(').replace(']', ')'))
kernel_size_down = eval(repr(net_config['kernel_size_down']).replace('[', '(').replace(']', ')'))
kernel_size_up = eval(repr(net_config['kernel_size_up']).replace('[', '(').replace(']', ')'))


class MtlsdModel(torch.nn.Module):

    def __init__(
            self,
            stack_infer=False,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            output_shapes=output_shapes,
        ):

        super().__init__()

        self.stack_infer = stack_infer

        self.unet = UNet(
                in_channels=in_channels,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                downsample_factors=downsample_factors,
                kernel_size_down=kernel_size_down,
                kernel_size_up=kernel_size_up,
                constant_upsample=True,
                padding="valid")

        self.lsd_head = ConvPass(num_fmaps, output_shapes[0], [[1, 1]], activation='Sigmoid')
        self.aff_head = ConvPass(num_fmaps, output_shapes[1], [[1, 1]], activation='Sigmoid')

    def forward(self, input):

        z = self.unet(input)

        lsds = self.lsd_head(z)
        affs = self.aff_head(z)
        

        if self.stack_infer: # add Z dimension during prediction
            lsds = torch.unsqueeze(lsds,-3)
            affs = torch.unsqueeze(affs,-3)

        return lsds, affs


class WeightedMSELoss(torch.nn.Module):

    def __init__(self):
        super(WeightedMSELoss, self).__init__()

    def _calc_loss(self, pred, target, weights):

        scale = (weights * (pred - target) ** 2)

        if len(torch.nonzero(scale)) != 0:

            mask = torch.masked_select(scale, torch.gt(weights, 0))
            loss = torch.mean(mask)

        else:

            loss = torch.mean(scale)

        return loss

    def forward(
            self,
            lsds_prediction,
            lsds_target,
            lsds_weights,
            affs_prediction,
            affs_target,
            affs_weights):

        lsds_loss = self._calc_loss(lsds_prediction, lsds_target, lsds_weights)
        affs_loss = self._calc_loss(affs_prediction, affs_target, affs_weights)

        return lsds_loss + affs_loss
