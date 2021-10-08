import os

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model

import hyperparameters

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




def Light_SERNet_V1(output_class,
                    input_duration=hyperparameters.INPUT_DURATIONS,
                    frame_length=hyperparameters.FFT_LENGTH,
                    frame_step=hyperparameters.FRAME_STEP,
                    fs=hyperparameters.SAMPLE_RATE,
                    n_mfcc=hyperparameters.N_MFCC):



    number_of_frame = (input_duration * fs - frame_length + frame_step) // frame_step
    body_input = layers.Input(shape=(number_of_frame, n_mfcc, 1))

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