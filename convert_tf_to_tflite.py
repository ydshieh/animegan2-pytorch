import tensorflow as tf
import os


import sys
sys.path.append("../")
from model_tf import TFGenerator


def convert_to_saved_model(model_path, target_shape, output_path, **kwargs):

    assert model_path.endswith(".h5")

    build_for_tflite = kwargs.pop("build_for_tflite", False)
    align_corners = kwargs.pop("align_corners", True)

    # create the model and load the weights
    model = TFGenerator(build_for_tflite=build_for_tflite)

    img_tf = tf.zeros(shape=target_shape)
    # make sure tf model is built
    model(img_tf, align_corners=align_corners)

    model.load_weights(model_path)
    model.save(output_path)

    # Make sure the loaded saved model works.
    _ = tf.keras.models.load_model(output_path)


def convert_to_tflite(saved_model_path, output_path):

    # TFLite
    # Convert the model
    # path to the SavedModel directory
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    # # Ensure that if any ops can't be quantized, the converter throws an error
    # # This requires a representative_dataset.
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    converter.target_spec.supported_ops = [
        # enable TensorFlow Lite ops
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]

    # # Set the input and output tensors to uint8 (APIs added in r2.3)
    # #   - Got: ValueError: The inference_input_type and inference_output_type must be tf.float32.
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()

    # Save the model.
    with open(output_path, 'wb') as f:
        f.write(tflite_model)


if __name__ == "__main__":

    p = "./weights/"
    pretrained_name = "face_paint_512_v2"

    height, width = (512, 512)
    target_shape = (1, 3, height, width,)

    model_path = os.path.join(p, f"{pretrained_name}_tf.h5")
    output_path = os.path.join(p, f"{pretrained_name}_saved_model")
    convert_to_saved_model(model_path, target_shape, output_path)

    model_path = os.path.join(p, f"{pretrained_name}_tf_lite.h5")
    output_path = os.path.join(p, f"{pretrained_name}_tf_lite_saved_model")
    convert_to_saved_model(model_path, target_shape, output_path, build_for_tflite=True)

    tflite_output_path = os.path.join(p, f"{pretrained_name}.tflite")
    convert_to_tflite(saved_model_path=output_path, output_path=tflite_output_path)
