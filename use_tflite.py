import tensorflow as tf
from PIL import Image
import time
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
interpreter.invoke()
stop_time = time.time()