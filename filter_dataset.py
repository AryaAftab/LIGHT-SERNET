import os
from glob import glob
from tqdm import tqdm

import tensorflow as tf

import hyperparameters


#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def seperate_iemocap_class(dataset_folder_name, target_classes=['angry', 'neutral', 'sadness'], merge_classes=['happiness', 'excited']):

    new_folder_name = "_".join(merge_classes)
    new_folder_path = f"{hyperparameters.BASE_DIRECTORY}/{dataset_folder_name}/{new_folder_name}"
    os.makedirs(new_folder_path, exist_ok=True)
    
    for sub_class in merge_classes:
        buff_path = f"{hyperparameters.BASE_DIRECTORY}/{dataset_folder_name}/{sub_class}/*.wav"
        filenames = glob(buff_path)

        print(f"Merge Phase Class : {sub_class}")
        with tqdm(total=len(filenames), position=0, leave=True) as pbar:
            for filename in tqdm(filenames , position=0, leave=True):
                os.system('cp {} {}'.format(filename, new_folder_path))
    
    
    target_classes.append(new_folder_name)
    non_target_classes = set(os.listdir(f"{hyperparameters.BASE_DIRECTORY}/{dataset_folder_name}")) - set(target_classes)

    print("Delete Useless Class Phase : ")
    with tqdm(total=len(non_target_classes), position=0, leave=True) as pbar:
        for buff in tqdm(non_target_classes , position=0, leave=True):
            folder_path = f"{hyperparameters.BASE_DIRECTORY}/{dataset_folder_name}/{buff}"
            os.system('rm -rf {}'.format(folder_path))
    
    print("Merge Done!")




def filter_iemocap(filenames, audio_type='None'):
    '''
    audio type = {
      None : for all data,
      impro : for improsived part,
      script : for scripted part
    }
    '''
    
    if audio_type is 'None':
        return tf.stack(filenames)
    
    buff_filenames = []
    for filename in filenames:
        if audio_type in str(filename):
            buff_filenames.append(filename)
    
    return tf.stack(buff_filenames)