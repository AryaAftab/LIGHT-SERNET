import os

import tensorflow as tf
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_audios):

	# Initialize the interpreter
	interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()[0]
	output_details = interpreter.get_output_details()[0]

	predictions = np.zeros((len(test_audios),), dtype=int)
	for i, test_audio in enumerate(test_audios):

	    # Check if the input type is quantized, then rescale input data to uint8
	    if input_details['dtype'] == np.uint8:
	        input_scale, input_zero_point = input_details["quantization"]
	        test_audio = test_audio / input_scale + input_zero_point

	    test_audio = np.expand_dims(test_audio, axis=0).astype(input_details["dtype"])
	    interpreter.set_tensor(input_details["index"], test_audio)
	    interpreter.invoke()
	    output = interpreter.get_tensor(output_details["index"])[0]

	    predictions[i] = output.argmax()

	return predictions




# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type, test_audios, test_labels):

	predictions = run_tflite_model(tflite_file, test_audios)

	accuracy = (np.sum(test_labels == predictions) * 100) / len(test_audios)

	print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
	    model_type, accuracy, len(test_audios)))