import torch
import tensorflow as tf


from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

import sys
sys.path.append("../")
from model_tf import TFGenerator

device = "cpu"

model_pt_state_dict = torch.load("./weights/model_pt_state_dict", map_location=device)

# ###########################
# # Test
# from model_pt import PTGenerator
#
# model = PTGenerator()
# model = model.eval().to(device)
# model.load_state_dict(torch.load("./weights/model_pt_state_dict", map_location=device))
# model_state_dict = model.state_dict()
#
#
# def face2paint(
#         img: Image.Image,
#         side_by_side: bool = False,
# ) -> Image.Image:
#     img_pt = to_tensor(img).unsqueeze(0) * 2 - 1
#     output = model(img_pt.to(device)).cpu()[0]
#
#     if side_by_side:
#         output = torch.cat([input[0], output], dim=2)
#
#     output = (output * 0.5 + 0.5).clip(0, 1)
#
#     return to_pil_image(output)
#
#
# # Inference
# img_path = "./examples/dasha-taran-4-cropped.jpg"
# width, height = (512, 512)
# channels = 3
#
# with Image.open(img_path) as img:
#     shape = (width, height)
#     img_resized = img.resize(shape)
#
#     animed_img = face2paint(img_resized)
#
#     img_resized.show()
#     animed_img.show()
#
# exit(0)
# ###########################


model_tf = TFGenerator()
img_tf = tf.zeros(shape=(1, 3, 512, 512,))
model_tf(img_tf, align_corners=False)
model_tf.load_weights("./weights/model_tf")

from collections import OrderedDict
import json
original_pt_to_pt_correspondence = OrderedDict()


def convert_tf_name_to_pt_name(tf_name):

    tf_name = tf_name.replace("_._", "/")
    tf_name = tf_name.replace("kernel:0", "weight")
    tf_name = tf_name.replace("bias:0", "bias")
    tf_name = tf_name.replace("gamma:0", "weight")
    tf_name = tf_name.replace("beta:0", "bias")
    tf_name = '.'.join(tf_name.split("/")[1:])

    return tf_name


for weight in model_tf.weights:
    print(weight.name)
    print(convert_tf_name_to_pt_name(weight.name))
    print('----------------------')


tf_weight_value_tuples = []

print(len(model_tf.weights))
print(len(model_pt_state_dict))


for tf_weight in model_tf.weights:
    print(tf_weight.name)
    print(tf_weight.shape)
    print(len(tf_weight.shape))


for tf_weight, pt_name in zip(model_tf.weights, model_pt_state_dict):

    pt_k = convert_tf_name_to_pt_name(tf_weight.name)

    print(tf_weight.name)
    print(pt_k)
    print(pt_name)
    assert pt_k == pt_name

    pt_weight = model_pt_state_dict[pt_k]
    print(tf_weight.shape)
    print(pt_weight.shape)

    pt_weight = tf.constant(pt_weight.detach().cpu().numpy())
    if len(pt_weight.shape) == 4:
        pt_weight = tf.transpose(pt_weight, perm=(2, 3, 1, 0))
    else:
        pt_weight = tf.transpose(pt_weight)
    assert pt_weight.shape == tf_weight.shape

    tf_weight_value_tuples.append((tf_weight, pt_weight))

    print('----------------')

from tensorflow.python.keras import backend as K
K.batch_set_value(tf_weight_value_tuples)

model_tf(img_tf, align_corners=False)
model = model_tf
model.save_weights("./weights/model_tf")
model.load_weights("./weights/model_tf")

###########################
# Test


def face2paint(
        img: Image.Image,
        side_by_side: bool = False,
) -> Image.Image:
    img_pt = to_tensor(img).unsqueeze(0) * 2 - 1
    img_tf = tf.constant(img_pt.detach().cpu().numpy())

    output = model(img_tf, align_corners=False)

    # Change (B, H, W, C) to (B, C, H, W)
    output = tf.transpose(output, perm=(0, 3, 1, 2))

    # Take the 1st image
    output = output[0]

    output = torch.tensor(output.numpy()).to(device)

    if side_by_side:
        output = torch.cat([input[0], output], dim=2)

    output = (output * 0.5 + 0.5).clip(0, 1)

    return to_pil_image(output)


# Inference
img_path = "./examples/dasha-taran-4-cropped.jpg"
width, height = (512, 512)
channels = 3

with Image.open(img_path) as img:
    shape = (width, height)
    img_resized = img.resize(shape)

    animed_img = face2paint(img_resized)

    img_resized.show()
    animed_img.show()


for variable in model.weights:
    print(variable.name)
