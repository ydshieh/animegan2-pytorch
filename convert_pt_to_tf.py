import os
import sys
import tensorflow as tf
import torch

from PIL import Image
from tensorflow.python.keras import backend as K
from torchvision.transforms.functional import to_tensor, to_pil_image

from model_tf import TFGenerator

sys.path.append("../")


def convert_pt_to_tf(pt_weight_path, output_path, build_for_tflite=False):

    state_dict = torch.load(pt_weight_path, map_location="cpu")

    model_tf = TFGenerator(build_for_tflite=build_for_tflite)
    img_tf = tf.zeros(shape=(1, 3, 512, 512,))
    # make sure tf model is built
    model_tf(img_tf, align_corners=False)

    load_pytorch_state_dict_to_tf_weights(model_tf.weights, state_dict)

    model_tf.save_weights(output_path)


def load_pytorch_state_dict_to_tf_weights(tf_weights, state_dict):

    pt_weight_names = set(state_dict.keys())

    tf_weight_value_tuples = []

    for tf_weight in tf_weights:

        converted_pt_weight_name = convert_tf_weight_name_to_pt_weight_name(tf_weight.name)

        # For our simple models, there should be a complete correspondence
        assert converted_pt_weight_name in pt_weight_names
        pt_weight_names.remove(converted_pt_weight_name)
        pt_weight = state_dict[converted_pt_weight_name]
        pt_weight = tf.constant(pt_weight.detach().cpu().numpy())

        # require transpose for pt/tf weight conversion
        if len(pt_weight.shape) == 4:
            # related to convolution layers
            if "depthwise_kernel" in tf_weight.name:
                #    PT: Conv2D(groups=groups): (out_ch, int_ch/groups, k0, k1)
                # -> TF: DepthwiseConv2D: (self.kernel_size + (int_ch, self.depth_multiplier))
                # (our implementation will have `DepthwiseConv2D` only when `groups==int_ch`, and have `out_ch==int_ch`)
                pt_weight = tf.transpose(pt_weight, perm=(2, 3, 0, 1))
            else:
                pt_weight = tf.transpose(pt_weight, perm=(2, 3, 1, 0))
        else:
            # simple cases: dense layers, etc.
            pt_weight = tf.transpose(pt_weight)

        assert pt_weight.shape == tf_weight.shape
        tf_weight_value_tuples.append((tf_weight, pt_weight))

    # For our simple models, there should be a complete correspondence
    assert len(pt_weight_names) == 0

    K.batch_set_value(tf_weight_value_tuples)


def convert_tf_weight_name_to_pt_weight_name(tf_weight_name):

    tf_weight_name = tf_weight_name.replace("_._", "/")
    tf_weight_name = tf_weight_name.replace("depthwise_kernel:0", "weight")
    tf_weight_name = tf_weight_name.replace("kernel:0", "weight")
    tf_weight_name = tf_weight_name.replace("bias:0", "bias")
    tf_weight_name = tf_weight_name.replace("gamma:0", "weight")
    tf_weight_name = tf_weight_name.replace("beta:0", "bias")
    tf_weight_name = '.'.join(tf_weight_name.split("/")[1:])

    return tf_weight_name


if __name__ == "__main__":

    width, height = (512, 512)
    channels = 3

    p = "./weights/"
    pretrained_name = "face_paint_512_v2"
    pt_weight_path = os.path.join(p, f"{pretrained_name}_pt.pt")
    output_path = os.path.join(p, f"{pretrained_name}_tf.h5")
    output_path_1 = os.path.join(p, f"{pretrained_name}_tf_lite.h5")

    convert_pt_to_tf(pt_weight_path, output_path, build_for_tflite=False)
    convert_pt_to_tf(pt_weight_path, output_path_1, build_for_tflite=True)

    # create the model and load the weights
    model = TFGenerator(build_for_tflite=False)
    model_1 = TFGenerator(build_for_tflite=True)

    img_tf = tf.zeros(shape=(1, channels, height, width,))

    # make sure tf model is built
    model(img_tf, align_corners=True)
    model_1(img_tf, align_corners=True)

    model.load_weights(output_path)
    model_1.load_weights(output_path_1)

    def face2paint(
        model,
        img: Image.Image,
        side_by_side: bool = False,
    ) -> Image.Image:
        img_pt = to_tensor(img).unsqueeze(0) * 2 - 1
        img_tf = tf.constant(img_pt.detach().cpu().numpy())

        output = model(img_tf, align_corners=True)

        # Change (B, H, W, C) to (B, C, H, W)
        output = tf.transpose(output, perm=(0, 3, 1, 2))

        # Take the 1st image
        output = output[0]

        output = torch.tensor(output.numpy()).to("cpu")

        if side_by_side:
            output = torch.cat([img_pt[0], output], dim=2)

        output = (output * 0.5 + 0.5).clip(0, 1)

        return to_pil_image(output)

    # Inference
    img_path = "./examples/dasha-taran-4-cropped.jpg"

    with Image.open(img_path) as img:
        shape = (width, height)
        resized_img = img.resize(shape)

        output_img = face2paint(model, resized_img, side_by_side=True)
        output_img.show()

        output_img = face2paint(model_1, resized_img, side_by_side=True)
        output_img.show()

    # for weight in model.weights:
    #     print(weight.name)
    #
    # for weight in model_1.weights:
    #     print(weight.name)
