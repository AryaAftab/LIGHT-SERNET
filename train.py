import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import datetime
import time
import argparse

import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


from dataio import *
from callbacks import *
from model_saver import *
from loss import *
from tflite_evaluate import *
import hyperparameters
import models


import warnings
warnings.filterwarnings("ignore")


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

tf.random.set_seed(hyperparameters.SEED)
np.random.seed(hyperparameters.SEED)






# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-dn", "--dataset_name",
                required=True,
                type=str,
                help="dataset name")

ap.add_argument("-id", "--input_durations",
                required=True,
                type=float,
                help="input durations(sec)")

ap.add_argument("-at", "--audio_type",
                default="all",
                type=str,
                help="auido type to filter dataset(IEMOCAP)")

ap.add_argument("-ln", "--loss_name",
                default="cross_entropy",
                type=str,
                help="cost function name for training")

ap.add_argument("-v", "--verbose",
                default=1,
                type=int,
                help="verbose for training bar")

ap.add_argument("-it", "--input_type",
                default="mfcc",
                type=str,
                help="type of input(mfcc, spectrogram, mel_spectrogram)")

ap.add_argument("-c", "--cache",
                default="disk",
                type=str,
                help="type of caching dataset(mfcc, spectrogram, mel_spectrogram)")

ap.add_argument("-m", "--merge_tflite",
                default=False,
                type=bool,
                help="do you want mfcc feature extractor to be merged in tflite model?")

args = vars(ap.parse_args())


dataset_name = args["dataset_name"]
input_durations = args["input_durations"]
audio_type = args["audio_type"]
loss_name = args["loss_name"]
verbose = args["verbose"]
input_type = args["input_type"]
cache = args["cache"]
merge_tflite = args["merge_tflite"]






print(".................................. Segment Dataset Started .......................................")
Segmented_datasetname_format = "{}_{:.1f}s_Segmented"

buff = Segmented_datasetname_format.format(dataset_name, input_durations)
buff = f"{hyperparameters.BASE_DIRECTORY}/{buff}"
if not os.path.exists(buff):
    os.system(f"python utils/segment/segment_dataset.py -dp data/{dataset_name} -ip utils/DATASET_INFO.json -d {dataset_name} -l {input_durations} -m 1")
dataset_name = Segmented_datasetname_format.format(dataset_name, input_durations)
print(".................................. Segment Dataset finished ......................................")






threshold = 0

Result = []
Reports = []
Predicted_targets = np.array([])
Actual_targets = np.array([])



Filenames, Splited_Index, Labels_list = split_dataset(dataset_name, audio_type=audio_type)



for counter in range (hyperparameters.K_FOLD):
    print(44 * "*", f"Fold : {counter + 1}", 44 * "*")
    now = datetime.datetime.now()
    print(f"Time : [{now.hour} : {now.minute} : {now.second}]")
    start_time = time.time()
    


    learningrate_scheduler = LearningRateScheduler()

    return_bestweight = BestModelWeights()
    
    callbacks = [learningrate_scheduler, return_bestweight]
    if verbose == 0:
        callbacks += [ShowProgress(hyperparameters.EPOCHS)]

    
    train_dataset, test_dataset = make_dataset(dataset_name=dataset_name,
                                               filenames=Filenames,
                                               splited_index=Splited_Index,
                                               labels_list=Labels_list,
                                               index_selection_fold=counter,
                                               cache=cache,
                                               merge_tflite=merge_tflite,
                                               input_type=input_type,
                                               maker=True)
    

    model = models.Light_SERNet_V1(len(Labels_list), input_durations, input_type)


    if loss_name == "cross_entropy":
    	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    elif loss_name == "focal":
    	loss = SparseCategoricalFocalLoss(gamma=hyperparameters.GAMMA)
    else:
    	raise ValueError('Loss name not valid!')


    
    
    model.compile(optimizer=tf.keras.optimizers.Adam(hyperparameters.LEARNING_RATE),
                  loss=loss,
                  metrics=['accuracy']) 
    
    
    #steps_per_epoch = (len(Filenames) - len(Splited_Index[counter])) // hyperparameters.BATCH_SIZE + 1
    history = model.fit(train_dataset,
                        #steps_per_epoch=steps_per_epoch,
                        epochs=hyperparameters.EPOCHS,
                        validation_data=test_dataset,
                        callbacks=callbacks,
                        verbose=verbose)
    
    
    
    
    #################### Save True and Predicted Label (Weight Precision : Float32) #########################
    buff = []
    buff.append(min(history.history["loss"]))
    buff.append(min(history.history["val_loss"]))
    buff.append(max(history.history["accuracy"]))
    buff.append(max(history.history["val_accuracy"]))

    if threshold < max(history.history["val_accuracy"]):
        History = history.history
        threshold = max(history.history["val_accuracy"])
        best_weights = model.get_weights()
        best_counter = counter

    Result.append(buff)


    BuffX = []
    BuffY = []
    for buff in test_dataset:
        BuffX.append(buff[0])
        BuffY.append(buff[1])
    BuffX = tf.concat(BuffX, axis=0).numpy()
    BuffY = tf.concat(BuffY, axis=0).numpy()
    Prediction = np.argmax(model.predict(BuffX), axis=1)

    Predicted_targets = np.append(Predicted_targets, Prediction)
    Actual_targets = np.append(Actual_targets, BuffY)

    print("Test Accuracy : ", accuracy_score(BuffY, Prediction))
    #########################################################################################################
    


    print("Time(sec) : ", time.time() - start_time)




###################################### prepare the test part related to the best model ##########################################
_, test_dataset = make_dataset(dataset_name=dataset_name,
                               filenames=Filenames,
                               splited_index=Splited_Index,
                               labels_list=Labels_list,
                               index_selection_fold=best_counter,
                               cache=cache,
                               merge_tflite=merge_tflite,
                               input_type=input_type,
                               maker=True)
BuffX = []
BuffY = []
for buff in test_dataset:
    BuffX.append(buff[0])
    BuffY.append(buff[1])
BuffX = tf.concat(BuffX, axis=0).numpy()
BuffY = tf.concat(BuffY, axis=0).numpy()
#################################################################################################################################


model.set_weights(best_weights)
###################### Save Best Model in keras format (Weight Precision : Float32) ###############################
best_modelname_keras = f"model/{dataset_name}_{loss_name}_float32.h5"
model.save(best_modelname_keras)

if merge_tflite:
    mfcc_extractor = MFCCExtractor(hyperparameters.NUM_MEL_BINS,
                                   hyperparameters.SAMPLE_RATE,
                                   hyperparameters.LOWER_EDGE_HERTZ,
                                   hyperparameters.UPPER_EDGE_HERTZ,
                                   hyperparameters.FRAME_LENGTH,
                                   hyperparameters.FRAME_STEP,
                                   hyperparameters.N_MFCC)

    merged_model = tf.keras.models.Sequential(
        [
        tf.keras.layers.Input(shape=(int(input_durations * hyperparameters.SAMPLE_RATE))),
        tf.keras.layers.Lambda(lambda x: tf.transpose(x, perm=[1, 0])),
        mfcc_extractor,
        model
        ]
    )
###################################################################################################################

###################### Save Best Model in tflite format (Weight Precision : Float32) ##############################
best_modelname_float32 = f"model/{dataset_name}_{loss_name}_float32.tflite"
save_float32(model, best_modelname_float32, merge_tflite=False)
evaluate_model(best_modelname_float32, "float32", BuffX, BuffY)

if merge_tflite and (input_type == "mfcc"):
    best_modelname_float32 = f"model/{dataset_name}_fuse_{loss_name}_float32.tflite"
    save_float32(merged_model, best_modelname_float32, merge_tflite=True)

print("..........................................................................................................")
###################################################################################################################

###################### Save Best Model in tflite format (Weight Precision : Float16) ##############################
best_modelname_float16 = f"model/{dataset_name}_{loss_name}_float16.tflite"
save_float16(model, best_modelname_float16, merge_tflite=False)
evaluate_model(best_modelname_float16, "float16", BuffX, BuffY)

if merge_tflite and (input_type == "mfcc"):
    best_modelname_float16 = f"model/{dataset_name}_fuse_{loss_name}_float16.tflite"
    save_float16(merged_model, best_modelname_float16, merge_tflite=True)

print("..........................................................................................................")
###################################################################################################################

###################### Save Best Model in tflite format (Weight Precision : Int8) #################################
best_modelname_int8 = f"model/{dataset_name}_{loss_name}_int8.tflite"
save_int8(model, best_modelname_int8, merge_tflite=False)
evaluate_model(best_modelname_int8, "int8", BuffX, BuffY)

if merge_tflite and (input_type == "mfcc"):
    best_modelname_int8 = f"model/{dataset_name}_fuse_{loss_name}_int8.tflite"
    save_int8(merged_model, best_modelname_int8, merge_tflite=True)

print("..........................................................................................................")
###################################################################################################################




############################# Plot Confusion Matrix (Weight Precision : Float32) ##################################
Report = classification_report(Actual_targets,
                               Predicted_targets,
                               target_names=list(Labels_list),
                               digits=4)

print(Report)
with open(f"result/{dataset_name}_{loss_name}_Report.txt", "w") as f:
    f.write(Report)


plt.figure(figsize=(15,10))
cm = confusion_matrix(Actual_targets, Predicted_targets, labels=range(len(Labels_list)))
plot_confusion_matrix(cm, list(Labels_list), normalize=False)
plt.savefig(f"result/{dataset_name}_{loss_name}_TotalConfusionMatrix.pdf", bbox_inches='tight')
#plt.show()

plt.figure(figsize=(15,10))
plot_confusion_matrix(cm, list(Labels_list), normalize=True)
plt.savefig(f"result/{dataset_name}_{loss_name}_TotalConfusionMatrixNormalized.pdf", bbox_inches='tight')
#plt.show()

##################################################################################################################