# import torch
import tensorflow as tf
# from torch import nn
import torch.nn.functional as F


class TFConvNormLReLU(tf.keras.layers.Layer):

    def __init__(self, out_ch, kernel_size=3, stride=1, padding=1, pad_mode="reflect", pad_constant=0, groups=1, bias=False, **kwargs):

        super().__init__(**kwargs)

        allowed_pad_modes = {
            "constant",
            "reflect",
            "symmetric",
            # "replication"
        }

        if pad_mode not in allowed_pad_modes:
            raise ValueError(f"`pad_model` must be one of {allowed_pad_modes}. Got {pad_mode} instead.")
        self.pad_mode = pad_mode

        self.pad_constant = pad_constant

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        if len(padding) == 2:
            # (Height, Width)
            padding = (padding[0], padding[0], padding[1], padding[1])
        assert len(padding) == 4
        self.padding = [
            [0, 0],  # batch dimension
            [padding[0], padding[1]],  # Height dimension: (top, bottom)
            [padding[2], padding[3]],  # Width dimension: (left, right)
            [0, 0],  # channel dimension
        ]

        # TF's version doesn't use `groups`.
        self.conv2d = tf.keras.layers.Conv2D(
            filters=out_ch,
            kernel_size=kernel_size,
            strides=stride,
            padding="valid",
            data_format="channels_last",
            groups=groups,
            activation=None,
            use_bias=bias,
            # Used by TF's version.
            # kernel_initializer=tf.keras.initializers.VarianceScaling(),
            name="conv2d"
        )

        # self.normalization = tfa.layers.GroupNormalization()
        # TODO 1: PT's version use `torch.nn.GroupNorm`. However, TF doesn't have it (only exists in TFA).
        # TODO 2: `LayerNormalization` has default `epsilon=1e-3`, but `torch.nn.GroupNorm` has default `eps=1e-5`.
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-05, center=True, scale=True, name="normalization")

        # OK
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2, name="activation")

    def call(self, inputs, training=False):

        padded_inputs = tf.pad(
            tensor=inputs,
            paddings=self.padding,
            mode=self.pad_mode,
            constant_values=self.pad_constant,
        )

        x = self.conv2d(padded_inputs)
        x = self.normalization(x)
        x = self.activation(x)

        return x


class TFInvertedResBlock(tf.keras.layers.Layer):

    def __init__(self, in_ch, out_ch, expansion_ratio=2, **kwargs):

        super().__init__(**kwargs)

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))

        # In TF 1.0 version, we always add this layer.
        # IN PT version, this condition is added.
        self.expansion = None
        if expansion_ratio != 1:
            self.expansion = TFConvNormLReLU(out_ch=bottleneck, kernel_size=1, padding=0, name="expansion")

        # dw
        self.bottleneck = TFConvNormLReLU(out_ch=bottleneck, groups=bottleneck, bias=True, name="bottleneck")

        # pw
        self.conv2d = tf.keras.layers.Conv2D(
            filters=out_ch,
            kernel_size=1,
            strides=1,
            padding="valid",
            data_format="channels_last",
            groups=1,
            activation=None,
            use_bias=False,
            name="conv2d"
        )

        # TODO 1: PT's version use `torch.nn.GroupNorm`. However, TF doesn't have it (only exists in TFA).
        # TODO 2: `LayerNormalization` has default `epsilon=1e-3`, but `torch.nn.GroupNorm` has default `eps=1e-5`.
        self.normalization = tf.keras.layers.LayerNormalization(epsilon=1e-05, center=True, scale=True, name="normalization")

    def call(self, inputs, training=False):

        x = inputs
        if self.expansion is not None:
            x = self.expansion(x)

        x = self.bottleneck(x)
        x = self.conv2d(x)
        x = self.normalization(x)

        if self.use_res_connect:
            x = inputs + x

        return x


# class Generator(nn.Module):
#     def __init__(self, ):
#         super().__init__()
#
#         self.block_a = nn.Sequential(
#             ConvNormLReLU(3, 32, kernel_size=7, padding=3),
#             ConvNormLReLU(32, 64, stride=2, padding=(0, 1, 0, 1)),
#             ConvNormLReLU(64, 64)
#         )
#
#         self.block_b = nn.Sequential(
#             ConvNormLReLU(64, 128, stride=2, padding=(0, 1, 0, 1)),
#             ConvNormLReLU(128, 128)
#         )
#
#         self.block_c = nn.Sequential(
#             ConvNormLReLU(128, 128),
#             InvertedResBlock(128, 256, 2),
#             InvertedResBlock(256, 256, 2),
#             InvertedResBlock(256, 256, 2),
#             InvertedResBlock(256, 256, 2),
#             ConvNormLReLU(256, 128),
#         )
#
#         self.block_d = nn.Sequential(
#             ConvNormLReLU(128, 128),
#             ConvNormLReLU(128, 128)
#         )
#
#         self.block_e = nn.Sequential(
#             ConvNormLReLU(128, 64),
#             ConvNormLReLU(64, 64),
#             ConvNormLReLU(64, 32, kernel_size=7, padding=3)
#         )
#
#         self.out_layer = nn.Sequential(
#             nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0, bias=False),
#             nn.Tanh()
#         )
#
#     def forward(self, input, align_corners=True):
#         out = self.block_a(input)
#         half_size = out.size()[-2:]
#         out = self.block_b(out)
#         out = self.block_c(out)
#
#         if align_corners:
#             out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
#         else:
#             out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
#         out = self.block_d(out)
#
#         if align_corners:
#             out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
#         else:
#             out = F.interpolate(out, scale_factor=2, mode="bilinear", align_corners=False)
#         out = self.block_e(out)
#
#         out = self.out_layer(out)
#         return out


class TFGenerator(tf.keras.layers.Layer):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.block_a = [
            TFConvNormLReLU(out_ch=32, kernel_size=7, padding=3, name=f"block_a_._{0}"),
            TFConvNormLReLU(out_ch=64, stride=2, padding=(0, 1, 0, 1), name=f"block_a_._{1}"),
            TFConvNormLReLU(out_ch=64, name=f"block_a_._{2}")
        ]

        self.block_b = [
            TFConvNormLReLU(out_ch=128, stride=2, padding=(0, 1, 0, 1), name=f"block_b_._{0}"),
            TFConvNormLReLU(out_ch=128, name=f"block_b_._{1}")
        ]

        self.block_c = [
            TFConvNormLReLU(out_ch=128, name=f"block_c_._{0}"),
            TFInvertedResBlock(in_ch=128, out_ch=256, expansion_ratio=2, name=f"block_c_._{1}"),
            TFInvertedResBlock(in_ch=256, out_ch=256, expansion_ratio=2, name=f"block_c_._{2}"),
            TFInvertedResBlock(in_ch=256, out_ch=256, expansion_ratio=2, name=f"block_c_._{3}"),
            TFInvertedResBlock(in_ch=256, out_ch=256, expansion_ratio=2, name=f"block_c_._{4}"),
            TFConvNormLReLU(out_ch=128, name=f"block_c_._{5}"),
        ]

        self.block_d = [
            TFConvNormLReLU(out_ch=128, name=f"block_d_._{0}"),
            TFConvNormLReLU(out_ch=128, name=f"block_d_._{1}")
        ]

        self.block_e = [
            TFConvNormLReLU(out_ch=64, name=f"block_e_._{0}"),
            TFConvNormLReLU(out_ch=64, name=f"block_e_._{1}"),
            TFConvNormLReLU(out_ch=32, kernel_size=7, padding=3, name=f"block_e_._{2}")
        ]

        self.out_layer = [
            tf.keras.layers.Conv2D(filters=3, kernel_size=1, strides=1, padding="valid", use_bias=False, name=f"out_layer_._{0}"),
            tf.keras.layers.Activation("tanh")
        ]

    def call(self, inputs, align_corners=True, training=False):

        # Transpose (B, C, H, W) to (B, H, W, C)
        inputs = tf.transpose(inputs, perm=(0, 2, 3, 1))

        x = inputs

        # out = self.block_a(input)
        for i, layer_module in enumerate(self.block_a):
            x = layer_module(x)

        # half_size = x.size()[-2:]

        # out = self.block_b(out)
        for i, layer_module in enumerate(self.block_b):
            x = layer_module(x)

        # out = self.block_c(out)
        for i, layer_module in enumerate(self.block_c):
            x = layer_module(x)

        if align_corners:
            # out = F.interpolate(out, half_size, mode="bilinear", align_corners=True)
            raise ValueError("")
        else:
            H, W = tf.shape(x)[1:3]
            H *= 2
            W *= 2
            x = tf.image.resize(x, size=(H, W), method="bilinear", name="...")

        # out = self.block_d(out)
        for i, layer_module in enumerate(self.block_d):
            x = layer_module(x)

        if align_corners:
            # out = F.interpolate(out, input.size()[-2:], mode="bilinear", align_corners=True)
            raise ValueError("")
        else:
            H, W = tf.shape(x)[1:3]
            H *= 2
            W *= 2
            x = tf.image.resize(x, size=(H, W), method="bilinear", name="...")

        # out = self.block_e(out)
        for i, layer_module in enumerate(self.block_e):
            x = layer_module(x)

        # out = self.out_layer(out)
        for i, layer_module in enumerate(self.out_layer):
            x = layer_module(x)

        return x


if __name__ == "__main__":

    generator = TFGenerator()

    print(generator)
