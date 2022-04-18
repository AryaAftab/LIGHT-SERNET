import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

import hyperparameters

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




def Light_SERNet_V1(output_class,
                    input_duration,
                    input_type="mfcc"):

    
    number_of_frame = (int(input_duration * hyperparameters.SAMPLE_RATE) - hyperparameters.FRAME_LENGTH + hyperparameters.FRAME_STEP) // hyperparameters.FRAME_STEP
    if input_type == "mfcc":
        number_of_feature = hyperparameters.N_MFCC
        number_of_channel = 1
    elif input_type == "spectrogram":
        number_of_feature = hyperparameters.NUM_SPECTROGRAM_BINS
        number_of_channel = 1
    elif input_type == "mel_spectrogram":
        number_of_feature = hyperparameters.NUM_MEL_BINS
        number_of_channel = 1
    else:
        raise ValueError('input_type not valid!')


    body_input = layers.Input(shape=(number_of_frame, number_of_feature, number_of_channel))

    path1 = layers.Conv2D(32, (11,1), padding="same", strides=(1,1))(body_input)
    path2 = layers.Conv2D(32, (1, 9), padding="same", strides=(1,1))(body_input)
    path3 = layers.Conv2D(32, (3, 3), padding="same", strides=(1,1))(body_input)

    path1 = layers.BatchNormalization()(path1)
    path2 = layers.BatchNormalization()(path2)
    path3 = layers.BatchNormalization()(path3)

    path1 = layers.ReLU()(path1)
    path2 = layers.ReLU()(path2)
    path3 = layers.ReLU()(path3)

    path1 = layers.AveragePooling2D(pool_size=2, padding="same")(path1)
    path2 = layers.AveragePooling2D(pool_size=2, padding="same")(path2)
    path3 = layers.AveragePooling2D(pool_size=2, padding="same")(path3)


    feature_extractor = tf.keras.layers.Concatenate(axis=-1)([path1, path2, path3])

    x = layers.Conv2D(64, (3,3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(feature_extractor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)

    x = layers.Conv2D(96, (3,3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(2,2), padding="same")(x)


    x = layers.Conv2D(128, (3,3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(2,1) , padding="same")(x)

    x = layers.Conv2D(160, (3,3), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.AveragePooling2D(pool_size=(2,1) , padding="same")(x)

    x = layers.Conv2D(320, (1,1), strides=1, padding='same', use_bias=False, kernel_regularizer=regularizers.l2(hyperparameters.L2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)


    x = layers.Dropout(hyperparameters.DROPOUT)(x)


    body_output = layers.Dense(output_class, activation="softmax")(x)
    body_model = Model(inputs=body_input, outputs=body_output)

    return body_model



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