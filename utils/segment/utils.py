import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import json
import os
from glob import glob
import sys


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



def read_json(directory):
    ## This function read .json file
    #
    # input : 
    #   directory : directory of .json file
    # output : 
    #   content : content of .json file

    with open(directory) as f:
        content = json.load(f)
    return content



def make_segment_directory(dataset_info, segment_length, segment_mode=1):
    ## Creates directory tree's for the segmented dataset
    #
    # input : 
    #   dataset_info : A dictionary that contains dataset information.
    #   segment_length : length of segment part (seconds)
    #   segment_mode : (0 : each audio semgment to multi parts, 1 : each audio segment to one part)
    try:
        classes = dataset_info["Class"].values()
    except:
        classes = dataset_info["Class"]

    directory = dataset_info["Directory"]

    directory = directory.split(os.path.sep)
    if segment_mode in [0, 1]:
        directory[-1] += f"_{segment_length}s_Segmented"
        directory = "/".join(directory)
    else:
        raise Exception("Sorry, segment_mode must 0 or 1")

    for name in classes:
        new_name = os.path.join(directory, name)
        os.makedirs(new_name, exist_ok=True)

    return directory



def zero_padding(data, segment_length=48000):
    ## This function zero-padding  the data back and forth; To reach the desired size
    #
    # inputs :
    #   data : input data (shape(len_data,)) and must len_data <= segment_length
    #   segment_length : desired size (sample)
    # output : 
    #   padded_data : data with desired size

    padding = segment_length - np.shape(data)[0]
    pre = int(np.floor(padding/2))
    pos = int(-np.floor(-padding/2))
    padded_data = np.concatenate([np.zeros(pre, dtype=np.float32), data, np.zeros(pos, dtype=np.float32)], axis=0)
    return padded_data



def normalize(data, segment_length=48000):
    ## This function performs the normalization operation ([- 1,1]) of the data
    #
    # input :
    #   data : input data
    #   segment_length : length of segment part (sample)
    # output :
    #   normalized_samples : normalized data

    EPS = np.finfo(float).eps
    samples_99_percentile = np.percentile(np.abs(data), 99.9)
    normalized_samples = data / (samples_99_percentile + EPS)
    normalized_samples = np.clip(normalized_samples, -1, 1)
    normalized_samples = zero_padding(normalized_samples, segment_length=segment_length)
    return normalized_samples



def trim_wave(data, segment_length=48000, segment_mode=1):
    ## This function trims the data to length n * segment_length from front and back
    #
    # inputs :
    #   data : input data
    #   segment_length : length of segment part (sample)
    #   segment_mode : (0 : each audio semgment to multi parts, 1 : each audio segment to one part)
    # output :
    #   trimed_data : trimed data

    if data.shape[0] <= segment_length:
        return data

    if segment_mode == 0:
    	trim_size = data.shape[0] % segment_length
    	pre = int(np.floor(trim_size/2))
    	pos = int(-np.floor(-trim_size/2))
	
    	trimed_data = data[pre:-pos]
    	return trimed_data
    elif segment_mode == 1:
    	trim_size = data.shape[0] - segment_length
    	pre = int(np.floor(trim_size/2))
    	pos = int(-np.floor(-trim_size/2))
	
    	trimed_data = data[pre:-pos]
    	return trimed_data
    else:
    	raise Exception("Sorry, segment_mode must 0 or 1")



def read_wave(wave_path):
    ## This function reads data from .wav file
    #
    # input : 
    #   wave_path : path of .wav file
    # output :
    #   raw_data : readed data
    #   fs : sample rate

    audio_binary = tf.io.read_file(wave_path)
    raw_data, fs = tf.audio.decode_wav(audio_binary)
    raw_data = np.squeeze(raw_data.numpy())
    return raw_data, fs.numpy()


def cleaning_directory_filename(directory):
    ## This function clean dataset filenames
    #
    # input : 
    #   directory : directory of dataset

    folders = os.listdir(directory)
    for folder in folders:
        filenames = glob(os.path.join(directory, f"{folder}/*.wav"))
        clean_filenames = shuffle(filenames.copy())
        for filename, clean_filename in zip(filenames, clean_filenames):
            buff = clean_filename.split("/")
            buff[-1] = buff[-1].replace("_", "")
            clean_filename = "/".join(buff)
            os.rename(filename, clean_filename)


def write_wave(data, wave_path, fs):
    ## This function writes data to the wav_path file
    #
    # input : 
    #   data : input raw data
    #   wave_path : path for write data
    #   fs : sample rate

    data = np.expand_dims(data, axis=-1)
    encoded_data = tf.audio.encode_wav(data, sample_rate=fs)
    tf.io.write_file(wave_path, encoded_data)