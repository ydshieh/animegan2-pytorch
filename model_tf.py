import tensorflow as tf
import tensorflow_addons as tfa


class TFConvNormLReLU(tf.keras.layers.Layer):

    def __init__(
        self,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        pad_mode="reflect",
        groups=1,
        bias=False,
        pad_constant=0,
        build_for_tflite=False,
        **kwargs
    ):

        super().__init__(**kwargs)
        self.groups = groups
        self.build_for_tflite = build_for_tflite

        allowed_pad_modes = {
            "constant",
            # "replication",
            "reflect",
            "symmetric",
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

        if groups == 1 or not build_for_tflite:

            self.conv2d = tf.keras.layers.Conv2D(
                filters=out_ch,
                kernel_size=kernel_size,
                strides=stride,
                padding="valid",
                data_format="channels_last",
                groups=groups,  # ...
                activation=None,
                use_bias=bias,
                name="conv2d",
            )

        else:

            # TFLite doesn't support groups > 1 for `Conv2D` layer.
            # (https://github.com/tensorflow/tensorflow/issues/40044)
            # Use `DepthwiseConv2D` as a workaround for using this model in TFLite.
            # However, this only covers the case where `in_ch == groups`.

            if out_ch % groups > 0:
                raise ValueError(
                    "`out_ch` has to be divisible by `groups`. Got {out_ch} for `out_ch` and {groups} for `groups`"
                    "instead."
                )
            depth_multiplier = out_ch // groups

            self.conv2d = tf.keras.layers.DepthwiseConv2D(
                kernel_size=kernel_size,
                strides=stride,
                padding="valid",
                depth_multiplier=depth_multiplier,
                data_format="channels_last",
                activation=None,
                use_bias=bias,
                name="conv2d",
            )

        # 1. PT's version use `torch.nn.GroupNorm`. However, TF doesn't have it (only exists in TFA).
        # 2. `tf.keras.layers.LayerNormalization` will not match a Group Normalization layer with group size set to 1.
        #    See [https://www.tensorflow.org/api_docs/python/tf/keras/layers/LayerNormalization].
        # 3. The following documentation in `tfa.layers.GroupNormalization` seems to be incorrect:
        #        ... Relation to Layer Normalization: If the number of groups is set to 1, then this operation becomes
        #            identical to Layer Normalization.
        #    [https://www.tensorflow.org/addons/api_docs/python/tfa/layers/GroupNormalization]
        # 4. `tfa.layers.GroupNormalization` has default `epsilon=1e-3`, but `torch.nn.GroupNorm` has default
        #    `eps=1e-5`.
        self.normalization = tfa.layers.GroupNormalization(
            groups=1, epsilon=1e-5, center=True, scale=True, axis=-1, name="normalization"
        )

        # OK
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2, name="activation")

    def build(self, input_shape):

        # A sanity check
        if self.build_for_tflite and self.groups != 1:
            in_ch = input_shape[-1]
            if self.groups != in_ch:
                raise ValueError(
                    "If the model is built for TFLite, `self.groups` has to be equal to the number of input channels."
                    "Got {self.groups} for {self.groups} and {in_ch} for the number of input channels instead."
                )

    def call(self, inputs, training=False):

        padded_inputs = tf.pad(
            tensor=inputs,
            paddings=self.padding,
            mode=self.pad_mode,
            constant_values=self.pad_constant,
        )

        x = self.conv2d(padded_inputs, training=training)
        x = self.normalization(x, training=training)
        x = self.activation(x, training=training)

        return x


class TFInvertedResBlock(tf.keras.layers.Layer):

    def __init__(self, out_ch, expansion_ratio=2, build_for_tflite=False, **kwargs):

        super().__init__(**kwargs)

        self.out_ch = out_ch
        self.expansion_ratio = expansion_ratio
        self.build_for_tflite = build_for_tflite

        # pw
        self.conv2d = tf.keras.layers.Conv2D(
            filters=out_ch,
            kernel_size=1,
            padding="valid",
            data_format="channels_last",
            use_bias=False,
            name="conv2d"
        )

        self.normalization = tfa.layers.GroupNormalization(
            groups=1, epsilon=1e-5, center=True, scale=True, axis=-1, name="normalization"
        )

    def build(self, input_shape):

        in_ch = input_shape[-1]

        self.use_res_connect = in_ch == self.out_ch
        bottleneck = int(round(in_ch * self.expansion_ratio))

        # In TF 1.0 version, we always add this layer.
        # IN PT version, this condition is added.
        self.expansion = None
        if self.expansion_ratio != 1:
            self.expansion = TFConvNormLReLU(
                out_ch=bottleneck, kernel_size=1, padding=0, build_for_tflite=self.build_for_tflite, name="expansion"
            )

        # dw
        self.bottleneck = TFConvNormLReLU(
            out_ch=bottleneck, groups=bottleneck, bias=True, build_for_tflite=self.build_for_tflite, name="bottleneck"
        )

    def call(self, inputs, training=False):

        x = inputs
        if self.expansion is not None:
            x = self.expansion(x, training=training)

        x = self.bottleneck(x, training=training)
        x = self.conv2d(x, training=training)
        x = self.normalization(x, training=training)

        if self.use_res_connect:
            x = inputs + x

        return x


class TFGenerator(tf.keras.Model):

    def __init__(self, build_for_tflite=False, **kwargs):

        super().__init__(**kwargs)

        self.block_a = [
            TFConvNormLReLU(
                out_ch=32, kernel_size=7, padding=3, build_for_tflite=build_for_tflite, name=f"block_a_._{0}"
            ),
            TFConvNormLReLU(
                out_ch=64, stride=2, padding=(0, 1, 0, 1), build_for_tflite=build_for_tflite, name=f"block_a_._{1}"
            ),
            TFConvNormLReLU(
                out_ch=64, build_for_tflite=build_for_tflite, name=f"block_a_._{2}"
            )
        ]

        self.block_b = [
            TFConvNormLReLU(
                out_ch=128, stride=2, padding=(0, 1, 0, 1), build_for_tflite=build_for_tflite, name=f"block_b_._{0}"
            ),
            TFConvNormLReLU(out_ch=128, build_for_tflite=build_for_tflite, name=f"block_b_._{1}")
        ]

        self.block_c = [
            TFConvNormLReLU(out_ch=128, build_for_tflite=build_for_tflite, name=f"block_c_._{0}"),
            TFInvertedResBlock(out_ch=256, expansion_ratio=2, build_for_tflite=build_for_tflite, name=f"block_c_._{1}"),
            TFInvertedResBlock(out_ch=256, expansion_ratio=2, build_for_tflite=build_for_tflite, name=f"block_c_._{2}"),
            TFInvertedResBlock(out_ch=256, expansion_ratio=2, build_for_tflite=build_for_tflite, name=f"block_c_._{3}"),
            TFInvertedResBlock(out_ch=256, expansion_ratio=2, build_for_tflite=build_for_tflite, name=f"block_c_._{4}"),
            TFConvNormLReLU(out_ch=128, build_for_tflite=build_for_tflite, name=f"block_c_._{5}"),
        ]

        self.block_d = [
            TFConvNormLReLU(out_ch=128, build_for_tflite=build_for_tflite, name=f"block_d_._{0}"),
            TFConvNormLReLU(out_ch=128, build_for_tflite=build_for_tflite, name=f"block_d_._{1}")
        ]

        self.block_e = [
            TFConvNormLReLU(out_ch=64, build_for_tflite=build_for_tflite, name=f"block_e_._{0}"),
            TFConvNormLReLU(out_ch=64, build_for_tflite=build_for_tflite, name=f"block_e_._{1}"),
            TFConvNormLReLU(
                out_ch=32, build_for_tflite=build_for_tflite, kernel_size=7, padding=3, name=f"block_e_._{2}"
            )
        ]

        self.out_layer = [
            tf.keras.layers.Conv2D(
                filters=3, kernel_size=1, strides=1, padding="valid", use_bias=False, name=f"out_layer_._{0}"
            ),
            tf.keras.layers.Activation("tanh")
        ]

    def call(self, inputs, align_corners=True, training=False):

        # Transpose (N, C, H, W) to (N, H, W, C)
        inputs = tf.transpose(inputs, perm=(0, 2, 3, 1))
        x = inputs

        for i, layer_module in enumerate(self.block_a):
            x = layer_module(x, training=training)

        half_size = tf.shape(x)[1:-1]

        for i, layer_module in enumerate(self.block_b):
            x = layer_module(x, training=training)

        for i, layer_module in enumerate(self.block_c):
            x = layer_module(x, training=training)

        # Use this way to avoid the following error when calling `model.save()`:
        #   - iterating over `tf.Tensor` is not allowed: AutoGraph did convert this function.
        #     This might indicate you are trying to use an unsupported feature.
        (height, width) = half_size[0], half_size[1]
        if not align_corners:
            shape = tf.shape(x)[1:-1]
            (height, width) = shape[0], shape[1]
            height *= 2
            width *= 2
        # TODO: how to achieve `align_corners`?
        x = tf.image.resize(x, size=(height, width), method="bilinear", name="resize")

        for i, layer_module in enumerate(self.block_d):
            x = layer_module(x, training=training)

        shape = tf.shape(inputs)[1:-1]
        (height, width) = shape[0], shape[1]
        if not align_corners:
            shape = tf.shape(x)[1:-1]
            (height, width) = shape[0], shape[1]
            height *= 2
            width *= 2
        # TODO: how to achieve `align_corners`?
        x = tf.image.resize(x, size=(height, width), method="bilinear", name="resize")

        for i, layer_module in enumerate(self.block_e):
            x = layer_module(x, training=training)

        for i, layer_module in enumerate(self.out_layer):
            x = layer_module(x, training=training)

        return x


if __name__ == "__main__":

    (N, C, H, W) = (1, 3, 64, 64)
    pixel_values = tf.zeros(shape=(N, C, H, W))

    generator = TFGenerator(build_for_tflite=False)
    print(generator)

    output = generator(pixel_values, align_corners=True)

    # for weight in generator.weights:
    #     print(weight.name)
    #     print(tf.shape(weight))
    #     print('--------------')

    print(tf.shape(output))

    generator = TFGenerator(build_for_tflite=False)
    print(generator)

    output = generator(pixel_values, align_corners=False)
    print(tf.shape(output))

    generator = TFGenerator(build_for_tflite=True)
    print(generator)

    output = generator(pixel_values)
    print(tf.shape(output))
