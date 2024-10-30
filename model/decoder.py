import torch
import torch.nn as nn
from einops import rearrange


def _make_resamplers(in_features, out_features):
    if isinstance(in_features, int):
        in_features = [in_features]*4
    resamplers = nn.ModuleList([])
    for level in range(4):
        in_channels = in_features[level]

        layers = []
        # 1x1 conv to convert n_channels
        layers.append(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_features[level],
                kernel_size=1,
                stride=1,
                padding=0,
            )
        )
        # upsample layers using deconvolution
        if level < 2:
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=out_features[level],
                    out_channels=out_features[level],
                    kernel_size=2**(2 - level),
                    stride=2**(2 - level),
                    padding=0,
                    bias=True,
                    dilation=1,
                    groups=1,
                ),
            )
        # downsample layers using convolution
        if level == 3:
            layers.append(
                nn.Conv2d(
                    in_channels=out_features[level],
                    out_channels=out_features[level],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            )
        resamplers.append(nn.Sequential(*layers))

    return resamplers


def _make_projectors(in_features, out_features):
    if isinstance(out_features, int):
        out_features = [out_features]*4
    projectors = nn.ModuleList([
        nn.Conv2d(
            in_features[i],
            out_features[i],
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        for i in range(4)
    ])

    return projectors


def _make_fusion_blocks(features, n_layers=1):
    fusion_blocks = nn.ModuleList([
        FeatureFusionBlock(
            features,
            nn.ReLU(False),
            deconv=False,
            expand=False,
            align_corners=True,
            n_layers=n_layers,
            upsample=True
        )
        for i in range(4)
    ])

    return fusion_blocks


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """
        
        cast = False
        if x.dtype != torch.float32:
            cast = True
            dtype = x.dtype
            x = x.float()

        if (x.numel() * self.scale_factor**2) > 2147483647:
            x_ = []
            for i in range(len(x)):
                x_.append(self.interp(
                    x[i:i+1],
                    scale_factor=self.scale_factor,
                    mode=self.mode,
                    align_corners=self.align_corners,
                ))
            x = torch.cat(x_)
        else:
            x = self.interp(
                x,
                scale_factor=self.scale_factor,
                mode=self.mode,
                align_corners=self.align_corners,
            )
        
        if cast:
            x = x.to(dtype)

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        n_layers=1,
        upsample=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.upsample = upsample

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        if n_layers == 1:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
            self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        else:
            self.resConfUnit1 = nn.Sequential(*[ResidualConvUnit(features, activation, bn)
                                                for _ in range(n_layers)])
            self.resConfUnit2 = nn.Sequential(*[ResidualConvUnit(features, activation, bn)
                                                for _ in range(n_layers)])

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = output + res

        output = self.resConfUnit2(output)

        if self.upsample:
            cast = False
            if output.dtype != torch.float32:
                cast = True
                dtype = output.dtype
                output = output.float()
            output = nn.functional.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
            )
            if cast:
                output = output.to(dtype)
        

        output = self.out_conv(output)

        return output
    

class DPTDecoder(nn.Module):
    '''
    DPT Convolutional Decoder for VTM
    '''
    def __init__(self,
            grid_size,
            in_features,
            hidden_features=[96, 192, 384, 768],
            out_features=256,
            out_chans=1,
            n_fusion_layers=1,
            kernel_sizes=[1, 1, 1, 1],
            img_size=(224, 224)
        ):
        super().__init__()
        self.grid_size = grid_size
        self.resamplers = _make_resamplers(in_features, hidden_features)
        self.projectors = _make_projectors(hidden_features, out_features)
        self.fusion_blocks = _make_fusion_blocks(out_features, n_layers=n_fusion_layers)

        head_upsample_factor = 2 * img_size[0] / 16 / grid_size[0]
        upsample_layer = Interpolate(scale_factor=head_upsample_factor, mode="bilinear", align_corners=True)

        self.head = nn.Sequential(
            nn.Conv2d(out_features, out_features // 2, kernel_size=3, stride=1, padding=1),
            upsample_layer,
            nn.Conv2d(out_features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, out_chans, kernel_size=1, stride=1, padding=0),
        )
        
        if len(kernel_sizes) == 1:
            kernel_sizes = kernel_sizes * 4
    
    def gating_parameters(self):
        for blk in self.fusion_blocks:
            if blk.fusion_gating:
                yield blk.beta

    def gating_parameter_names(self):
        names = []
        for i, blk in enumerate(self.fusion_blocks):
            if blk.fusion_gating:
                names.append(f'fusion_blocks.{i}.beta')
        
        return names
    
    def forward(self, features):
        B, T, N = features[0].shape[:3]

        for level in range(3, -1, -1):
            feat = features[level]

            feat = rearrange(feat, 'B T N (h w) d -> (B T N) d h w',
                             h=self.grid_size[0], w=self.grid_size[1])

            feat = self.resamplers[level](feat)
            feat = self.projectors[level](feat)

            if level == 3:
                feat = self.fusion_blocks[level](feat)
            else:
                feat = self.fusion_blocks[level](up_feat, feat)
            up_feat = feat

        out = self.head(feat)
        out = rearrange(out, '(B T N) C H W -> B T N C H W', B=B, T=T, N=N)

        return out
