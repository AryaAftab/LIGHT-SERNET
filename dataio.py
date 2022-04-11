import os

import tensorflow as tf
import numpy as np

import hyperparameters
from filter_dataset import *



#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



AUTOTUNE = tf.data.experimental.AUTOTUNE


linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(hyperparameters.NUM_MEL_BINS,
									                                hyperparameters.NUM_SPECTROGRAM_BINS,
									                                hyperparameters.SAMPLE_RATE,
									                                hyperparameters.LOWER_EDGE_HERTZ,
									                                hyperparameters.UPPER_EDGE_HERTZ)



def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)



def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
  
    return parts[-2]



def get_waveform_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label



def get_spectrogram(waveform):

    waveform = tf.cast(waveform, tf.float32)
    spectrogram = tf.signal.stft(
        waveform, frame_length=hyperparameters.FRAME_LENGTH, frame_step=hyperparameters.FRAME_STEP, fft_length=hyperparameters.FFT_LENGTH)

    spectrogram = tf.abs(spectrogram)

    return spectrogram



def get_mel_spectrogram(spectrogram):

    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)

    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    return log_mel_spectrogram



def get_mfcc(log_mel_spectrograms, clip_value=10):
    mfcc = mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :hyperparameters.N_MFCC]

    return tf.clip_by_value(mfcc, -clip_value, clip_value)



def get_label_id(label, labels_list):
    label_id = tf.argmax(tf.cast(label == labels_list, tf.int64))

    return label_id


class MFCCExtractor(tf.keras.layers.Layer):
    def __init__(self,
                    NUM_MEL_BINS,
                    SAMPLE_RATE,
                    LOWER_EDGE_HERTZ,
                    UPPER_EDGE_HERTZ,
                    FRAME_LENGTH,
                    FRAME_STEP,
                    N_MFCC,
                    **kwargs):
        super(MFCCExtractor, self).__init__(**kwargs)

        self.NUM_MEL_BINS = NUM_MEL_BINS
        self.SAMPLE_RATE = SAMPLE_RATE
        self.LOWER_EDGE_HERTZ = LOWER_EDGE_HERTZ
        self.UPPER_EDGE_HERTZ = UPPER_EDGE_HERTZ

        self.FRAME_LENGTH = FRAME_LENGTH
        self.FRAME_STEP = FRAME_STEP

        self.N_MFCC = N_MFCC


    def get_mfcc(self, waveform, clip_value=10):
        waveform = tf.cast(waveform, tf.float32)
        spectrogram = tf.raw_ops.AudioSpectrogram(input=waveform,
                                                    window_size=self.FRAME_LENGTH,
                                                    stride=self.FRAME_STEP,
                                                    magnitude_squared=True,
                                                    )


        mfcc = tf.raw_ops.Mfcc(spectrogram=spectrogram,
                                sample_rate=hyperparameters.SAMPLE_RATE,
                                upper_frequency_limit=hyperparameters.UPPER_EDGE_HERTZ,
                                lower_frequency_limit=hyperparameters.LOWER_EDGE_HERTZ,
                                filterbank_channel_count=hyperparameters.NUM_MEL_BINS,
                                dct_coefficient_count=hyperparameters.N_MFCC,
                                )

        return tf.clip_by_value(mfcc, -clip_value, clip_value)


    def call(self, inputs):
        outputs = self.get_mfcc(inputs)

        return tf.expand_dims(outputs, -1)


    def get_config(self):
        config = super(MFCCExtractor, self).get_config()
        config.update({
            "NUM_MEL_BINS": self.NUM_MEL_BINS,
            "SAMPLE_RATE": self.SAMPLE_RATE,
            "LOWER_EDGE_HERTZ": self.LOWER_EDGE_HERTZ,
            "UPPER_EDGE_HERTZ": self.UPPER_EDGE_HERTZ,
            "FRAME_LENGTH": self.FRAME_LENGTH,
            "FRAME_STEP": self.FRAME_STEP,
            "N_MFCC": self.N_MFCC,
        })
        return config


def get_input_and_label_id(audio, label, labels_list, input_type="mfcc", merge_tflite=False):
    label_id = get_label_id(label, labels_list)

    if input_type == "spectrogram":
        spectrogram = get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        return spectrogram, label_id
    elif input_type == "mel_spectrogram":
        spectrogram = get_spectrogram(audio)
        mel_spectrogram = get_mel_spectrogram(spectrogram)
        mel_spectrogram = tf.expand_dims(mel_spectrogram, -1)
        return mel_spectrogram, label_id
    elif input_type == "mfcc":
        if merge_tflite:
            mfcc = MFCCExtractor(hyperparameters.NUM_MEL_BINS,
                                 hyperparameters.SAMPLE_RATE,
                                 hyperparameters.LOWER_EDGE_HERTZ,
                                 hyperparameters.UPPER_EDGE_HERTZ,
                                 hyperparameters.FRAME_LENGTH,
                                 hyperparameters.FRAME_STEP,
                                 hyperparameters.N_MFCC)(audio[..., None])[0]
        else: 
            spectrogram = get_spectrogram(audio)
            mel_spectrogram = get_mel_spectrogram(spectrogram)
            mfcc = get_mfcc(mel_spectrogram)
            mfcc = tf.expand_dims(mfcc, -1)

        return mfcc, label_id
    else:
        raise ValueError('input_type not Valid!')



def preprocess_dataset(files, labels_list, input_type="mfcc", merge_tflite=False):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
    output_ds = output_ds.map(lambda x, y : 
        get_input_and_label_id(x, y, labels_list, input_type, merge_tflite),  num_parallel_calls=AUTOTUNE)

    return output_ds





def split_dataset(dataset_name, audio_type="all"):
    
    dataset_name = os.path.join(hyperparameters.BASE_DIRECTORY, dataset_name)
    
    if "IEMOCAP" in dataset_name:
        labels_list = np.array(tf.io.gfile.listdir(str(dataset_name)))
        if len(labels_list) != 4:
            seperate_iemocap_class(dataset_name,
                                   target_classes=['angry', 'neutral', 'sadness'],
                                   merge_classes=['happiness', 'excited'])
                
    
    filenames = tf.io.gfile.glob(str(dataset_name) + '/*/*')
    

    if "IEMOCAP" in dataset_name:
        filenames = filter_iemocap(filenames, audio_type=audio_type)
        filenames = tf.random.shuffle(filenames)
        num_samples = len(filenames)
        labels_list = np.array(tf.io.gfile.listdir(str(dataset_name)))
        splited_index = seperate_speaker_id_iemocap(filenames)
    else:
        filenames = tf.random.shuffle(filenames)
        num_samples = len(filenames)
        labels_list = np.array(tf.io.gfile.listdir(str(dataset_name)))
        splited_index = seperate_speaker_id_emodb(filenames)
        
    return filenames, splited_index, labels_list





def make_dataset(dataset_name, filenames, splited_index, labels_list, index_selection_fold, cache="disk", merge_tflite=False, input_type="mfcc", maker=True):

    if cache == "disk":
        cache_directory = f"{hyperparameters.BASE_DIRECTORY}/Cache/{dataset_name}"
        os.system(f"rm -rf {cache_directory}")

        train_cache_directory = os.path.join(cache_directory, "train")
        test_cache_directory = os.path.join(cache_directory, "test")

        os.makedirs(train_cache_directory, exist_ok=True)
        os.makedirs(test_cache_directory, exist_ok=True)


    test_index = splited_index[index_selection_fold]

    train_index = np.setdiff1d(np.arange(len(filenames)), test_index)


    train_files = tf.gather(filenames, train_index)
    test_files = tf.gather(filenames, test_index)


    train_dataset = preprocess_dataset(train_files, labels_list, input_type, merge_tflite)
    test_dataset = preprocess_dataset(test_files, labels_list, input_type, merge_tflite)
    if cache == "disk":
        train_dataset = train_dataset.cache(train_cache_directory + "/file")
        test_dataset = test_dataset.cache(test_cache_directory+ "/file")
    elif cache == "ram":
        train_dataset = train_dataset.cache()
        test_dataset = test_dataset.cache()
    elif cache== "None":
        pass
    else:
        raise ValueError('cache not Valid!')


    train_dataset = train_dataset.shuffle(len(train_files))
    train_dataset = train_dataset.batch(hyperparameters.BATCH_SIZE).prefetch(AUTOTUNE)
    
    test_dataset = test_dataset.batch(hyperparameters.BATCH_SIZE).prefetch(AUTOTUNE)

    if maker: 
        list(test_dataset.as_numpy_iterator()) 
        list(train_dataset.as_numpy_iterator()) 

    return train_dataset, test_dataset