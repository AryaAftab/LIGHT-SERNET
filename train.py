import os
import datetime
import time
import argparse
import itertools

import tensorflow as tf


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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

ap.add_argument("-at", "--audio_type",
                default="None",
                type=str,
                help="auido type to filter dataset(IEMOCAP)")

ap.add_argument("-ln", "--loss_name",
                default="cross_entropy",
                type=str,
                help="loss for training")

ap.add_argument("-v", "--verbose",
                default=1,
                type=int,
                help="verbose for training bar")

args = vars(ap.parse_args())


dataset_name = args["dataset_name"]
audio_type = args["audio_type"]
loss_name = args["loss_name"]
verbose = args["verbose"]







threshold = 0

Result = []
Reports = []
Predicted_targets = np.array([])
Actual_targets = np.array([])


Index_Selection_Fold = np.array(list(itertools.combinations(range(hyperparameters.K_FOLD),2)))
Index_Selection_Fold = np.random.permutation(Index_Selection_Fold)[:hyperparameters.K_FOLD]
Filenames, Splited_Index, Labels_list = split_dataset(dataset_name, audio_type=audio_type)



for counter in range (hyperparameters.K_FOLD):
    print(40 * "*", f"Fold : {counter + 1}", 40 * "*")
    now = datetime.datetime.now()
    print(f"Time : [{now.hour} : {now.minute} : {now.second}]")
    start_time = time.time()
    


    learningrate_scheduler = LearningRateScheduler()

    
    return_bestweight = ReturnBestEarlyStopping(
        monitor='val_accuracy',
        min_delta=0,
        patience=10000,
        verbose=0,
        mode='max',
        baseline=None,
        restore_best_weights=True
    )
    

    
    train_dataset, test_dataset = make_dataset_with_cache(dataset_name, Filenames, Splited_Index, Labels_list, Index_Selection_Fold[counter])


    model = models.Light_SERNet_V1(len(Labels_list))


    if loss_name == "cross_entropy":
    	loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    elif loss_name == "focal":
    	loss = SparseCategoricalFocalLoss(gamma=hyperparameters.GAMMA)
    else:
    	raise ValueError('Loss name not Valid!')


    
    
    model.compile(optimizer=tf.keras.optimizers.Adam(hyperparameters.LEARNING_RATE),
              loss=loss,
              metrics=['accuracy']) 
    
    
    history = model.fit(train_dataset,
                        steps_per_epoch=int(0.8 * len(Filenames) / hyperparameters.BATCH_SIZE),
                        epochs=hyperparameters.EPOCHS,
                        validation_data=test_dataset,
                        callbacks=[learningrate_scheduler, return_bestweight],
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
        best_model = tf.keras.models.clone_model(model)
        best_counter = counter


    Result.append(buff)
    print("Validation Accuracy : ", buff[3])

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
    #########################################################################################################
    


    print("Time(sec) : ", time.time() - start_time)




###################################### prepare the test part related to the best model ##########################################
_, test_dataset = make_dataset_with_cache(dataset_name, Filenames, Splited_Index, Labels_list, Index_Selection_Fold[best_counter])
BuffX = []
BuffY = []
for buff in test_dataset:
    BuffX.append(buff[0])
    BuffY.append(buff[1])
BuffX = tf.concat(BuffX, axis=0).numpy()
BuffY = tf.concat(BuffY, axis=0).numpy()
#################################################################################################################################

###################### Save Best Model in tflite format (Weight Precision : Float32) #########################
best_modelname_float32 = f"model/{dataset_name}_float32.tflite"
save_float32(best_model, best_modelname_float32)

evaluate_model(best_modelname_float32, "float32", BuffX, BuffY)
##############################################################################################################

###################### Save Best Model in tflite format (Weight Precision : Float16) #########################
best_modelname_float16 = f"model/{dataset_name}_float16.tflite"
save_float16(best_model, best_modelname_float16)

evaluate_model(best_modelname_float16, "float16", BuffX, BuffY)
##############################################################################################################

###################### Save Best Model in tflite format (Weight Precision : Int8) ############################
best_modelname_int8 = f"model/{dataset_name}_int8.tflite"
save_int8(best_model, best_modelname_int8)

evaluate_model(best_modelname_int8, "int8", BuffX, BuffY)
##############################################################################################################




########################## Plot Confusion Matrix (Weight Precision : Float32) ################################
Report = classification_report(Actual_targets,
                               Predicted_targets,
                               target_names=list(Labels_list),
                               digits=4)
print(Report)


plt.figure(figsize=(15,10))
cm = confusion_matrix(Actual_targets, Predicted_targets, labels=range(len(Labels_list)))
plot_confusion_matrix(cm, list(Labels_list), normalize=False)
plt.savefig(f"result/{dataset_name}_TotalConfusionMatrix.pdf", bbox_inches='tight')
plt.show()

plt.figure(figsize=(15,10))
plot_confusion_matrix(cm, list(Labels_list), normalize=True)
plt.savefig(f"result/{dataset_name}_TotalConfusionMatrixNormalized.pdf", bbox_inches='tight')
plt.show()

##############################################################################################################
