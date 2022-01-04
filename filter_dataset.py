import os
from glob import glob
from tqdm import tqdm

import numpy as np
import tensorflow as tf

import hyperparameters


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def seperate_iemocap_class(dataset_folder_path, target_classes=['angry', 'neutral', 'sadness'], merge_classes=['happiness', 'excited']):

    new_folder_name = "_".join(merge_classes)
    new_folder_path = f"{dataset_folder_path}/{new_folder_name}"
    os.makedirs(new_folder_path, exist_ok=True)
    
    for sub_class in merge_classes:
        buff_path = f"{dataset_folder_path}/{sub_class}/*.wav"
        filenames = glob(buff_path)

        print(f"Merge Phase Class : {sub_class}")
        with tqdm(total=len(filenames), position=0, leave=True) as pbar:
            for filename in tqdm(filenames , position=0, leave=True):
                os.system('cp {} {}'.format(filename, new_folder_path))
    
    
    target_classes.append(new_folder_name)
    non_target_classes = set(os.listdir(dataset_folder_path)) - set(target_classes)

    print("Delete Useless Class Phase : ")
    with tqdm(total=len(non_target_classes), position=0, leave=True) as pbar:
        for buff in tqdm(non_target_classes , position=0, leave=True):
            folder_path = f"{dataset_folder_path}/{buff}"
            os.system('rm -rf {}'.format(folder_path))
    
    print("Merge Done!")




def filter_iemocap(filenames, audio_type="all"):
    '''
    audio type = {
      all : for all data,
      impro : for improsived part,
      script : for scripted part
    }
    '''
    
    if audio_type == "all":
        return tf.stack(filenames)
    
    buff_filenames = []
    for filename in filenames:
        if audio_type in str(filename):
            buff_filenames.append(filename)
    
    return tf.stack(buff_filenames)



def seperate_speaker_id_emodb(filenames):
    list_of_speaker_ids = ["03", "08", "09", "10", "11", "12", "13", "14", "15", "16"]
    
    
    splited_index = []
    for speaker_id in list_of_speaker_ids:
        buff = []
        for counter, filename in enumerate(filenames):
            filename_id = os.path.basename(str(filename))[:2]
            if filename_id == speaker_id:
                buff.append(counter)
        splited_index.append(np.array(buff))

    
    return splited_index



def seperate_speaker_id_iemocap(filenames):
    
    speaker_format = "Ses0{}{}"
    
    list_of_speaker_ids = []
    for n_session in range(5):
        for sex in ["F", "M"]:
            list_of_speaker_ids.append(speaker_format.format(n_session + 1, sex))
    
    
    
    splited_index = []
    for speaker_id in list_of_speaker_ids:
        buff = []
        for counter, filename in enumerate(filenames):
            filename_id = os.path.basename(str(filename))
            filename_id = filename_id[:6]
            if filename_id == speaker_id:
                buff.append(counter)
        splited_index.append(np.array(buff))

    
    return splited_index