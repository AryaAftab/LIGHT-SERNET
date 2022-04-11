import os

import tensorflow as tf


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def save_float32(model, name, merge_tflite):
	float32_converter = tf.lite.TFLiteConverter.from_keras_model(model)
	if merge_tflite:
		float32_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
													   tf.lite.OpsSet.SELECT_TF_OPS]
	float32_tflite_model = float32_converter.convert()

	with open(name, 'wb') as f:
		f.write(float32_tflite_model)

	print("Float32 model in Mb:", os.path.getsize(name) / float(2**20))




def save_float16(model, name, merge_tflite):
	float16_converter = tf.lite.TFLiteConverter.from_keras_model(model)
	if merge_tflite:
		float16_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
													   tf.lite.OpsSet.SELECT_TF_OPS]
	float16_converter.optimizations = [tf.lite.Optimize.DEFAULT]
	float16_converter.target_spec.supported_types = [tf.float16]
	float16_tflite_model = float16_converter.convert()

	with open(name, 'wb') as f:
		f.write(float16_tflite_model)

	print("Float16 model in Mb:", os.path.getsize(name) / float(2**20))




def save_int8(model, name, merge_tflite):
	int8_converter = tf.lite.TFLiteConverter.from_keras_model(model)
	if merge_tflite:
		int8_converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
													   tf.lite.OpsSet.SELECT_TF_OPS]
	int8_converter.optimizations = [tf.lite.Optimize.DEFAULT]
	int8_tflite_model = int8_converter.convert()


	with open(name, 'wb') as f:
		f.write(int8_tflite_model)

	print("Int8 model in Mb:", os.path.getsize(name) / float(2**20))