import tensorflow as tf
from PIL import Image
import time
import datetime
from torchvision.transforms.functional import to_tensor

from PIL import Image
#import tflite_runtime.interpreter as tflite


#interpreter = tflite.Interpreter(model_path="tflite_model.tflite")
interpreter = tf.lite.Interpreter(model_path="tflite_model.tflite")

print(interpreter)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)




# Inference
img_path = "./examples/dasha-taran-4-cropped.jpg"
width, height = (512, 512)
channels = 3



with Image.open(img_path) as img:
    shape = (width, height)
    img_resized = img.resize(shape)

    img_pt = to_tensor(img_resized).unsqueeze(0) * 2 - 1
    img_tf = img_pt.detach().cpu().numpy()

input_data = img_tf


interpreter.set_tensor(input_details[0]['index'], input_data)

start_time = time.time()
print(datetime.datetime.now())

interpreter.invoke()

# stop_time = time.time()
# print(datetime.datetime.now())
# print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))

output_data = interpreter.get_tensor(output_details[0]['index'])

print(datetime.datetime.now())

stop_time = time.time()
print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))


import torch
from torchvision.transforms.functional import to_tensor, to_pil_image

device = "cpu"
# Change (B, H, W, C) to (B, C, H, W)
output = tf.transpose(output_data, perm=(0, 3, 1, 2))
# Take the 1st image
output = output[0]
output = torch.tensor(output.numpy()).to(device)
output = (output * 0.5 + 0.5).clip(0, 1)
animed_img = to_pil_image(output)
animed_img.show()


print(output_data)
