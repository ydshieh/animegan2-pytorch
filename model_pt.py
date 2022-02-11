from torch import nn
import torch.nn.functional as F


class PTConvNormLReLU(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", groups=1, bias=False):

        super().__init__()

        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise ValueError(f"`pad_model` must be one of {list(pad_layer.keys())}. Got {pad_mode} instead.")

        self.pad = pad_layer[pad_mode](padding)
        self.conv2d = nn.Conv2d(
            in_ch, out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            bias=bias,
        )
        self.normalization = nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True)
        self.activation = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, inputs):

        padded_inputs = self.pad(inputs)
        x = self.conv2d(padded_inputs)
        x = self.normalization(x)
        x = self.activation(x)

        return x


class PTInvertedResBlock(nn.Module):

    def __init__(self, in_ch, out_ch, expansion_ratio=2):

        super().__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))

        if expansion_ratio != 1:
            self.expansion = PTConvNormLReLU(in_ch=in_ch, out_ch=bottleneck, kernel_size=1, padding=0)
        else:
            self.expansion = None

        # dw
        self.bottleneck = PTConvNormLReLU(in_ch=bottleneck, out_ch=bottleneck, groups=bottleneck, bias=True)

        # pw
        self.conv2d = nn.Conv2d(in_channels=bottleneck, out_channels=out_ch, kernel_size=1, padding=0, bias=False)

        self.normalization = nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True)

    def forward(self, inputs):

        x = inputs
        if self.expansion is not None:
            x = self.expansion(x)

        x = self.bottleneck(x)
        x = self.conv2d(x)
        x = self.normalization(x)

        if self.use_res_connect:
            x = inputs + x

        return x


class PTGenerator(nn.Module):

    def __init__(self, ):

        super().__init__()

        self.block_a = nn.Sequential(
            PTConvNormLReLU(3, 32, kernel_size=7, padding=3),
            PTConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
            PTConvNormLReLU(64, 64)
        )

        self.block_b = nn.Sequential(
            PTConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
            PTConvNormLReLU(128, 128)
        )

        self.block_c = nn.Sequential(
            PTConvNormLReLU(128, 128),
            PTInvertedResBlock(128, 256, 2),
            PTInvertedResBlock(256, 256, 2),
            PTInvertedResBlock(256, 256, 2),
            PTInvertedResBlock(256, 256, 2),
            PTConvNormLReLU(256, 128),
        )

        self.block_d = nn.Sequential(
            PTConvNormLReLU(128, 128),
            PTConvNormLReLU(128, 128)
        )

        self.block_e = nn.Sequential(
            PTConvNormLReLU(128, 64),
            PTConvNormLReLU(64, 64),
            PTConvNormLReLU(64, 32, kernel_size=7, padding=3)
        )

        self.out_layer = nn.Sequential(
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, input, align_corners=True):

        out = self.block_a(input)

        half_size = out.size()[-2:]

        out = self.block_b(out)
        out = self.block_c(out)

        if align_corners:
            out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)

        out = self.block_d(out)

        if align_corners:
            out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
        else:
            out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)

        out = self.block_e(out)

        out = self.out_layer(out)
        return out


if __name__ == "__main__":

    import torch

    (N, C, H, W) = (1, 3, 64, 64)
    pixel_values = torch.zeros(size=(N, C, H, W))

    generator = PTGenerator()
    print(generator)

    for k, v in generator.state_dict().items():
        print(k)
        print(v.size())
        print('--------------')

    output = generator(pixel_values, align_corners=True)
    print(output.size())
