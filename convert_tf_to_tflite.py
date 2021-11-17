import torch
import tensorflow as tf

from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image

import sys
sys.path.append("../")
from model_tf import TFGenerator


torch.set_grad_enabled(False)
device = "cpu"

# model = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="face_paint_512_v2")
model = TFGenerator()

img_tf = tf.zeros(shape=(1, 3, 512, 512,))
model(img_tf, align_corners=False)
model.load_weights("./weights/model_tf")

print(model)


def face2paint(
        model: tf.keras.Model,
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

    animed_img = face2paint(model, img_resized)

    img_resized.show()
    animed_img.show()


model.save("./tf_saved_model")


# It can be used to reconstruct the model identically.
reconstructed_model = tf.keras.models.load_model("./tf_saved_model")
print(reconstructed_model)

with Image.open(img_path) as img:
    shape = (width, height)
    img_resized = img.resize(shape)

    animed_img = face2paint(reconstructed_model, img_resized)

    img_resized.show()
    animed_img.show()

### exit(0)


# TFLite
# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("./tf_saved_model")  # path to the SavedModel directory

# TO fix the normalization (`tf.nn.batch_normalization`) `op is neither a custom op nor a flex op` issue.
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
  # tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]

tflite_model = converter.convert()

# Save the model.
with open('tflite_model.tflite', 'wb') as f:
    f.write(tflite_model)


