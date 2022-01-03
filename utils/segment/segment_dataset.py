import os
import argparse
from tqdm import tqdm

from utils import *
from read_dataset import *

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-dp", "--dataset_path",
                required=True,
                type=str,
                help="path to dataset")

ap.add_argument("-ip", "--info_path",
                required=True,
                type=str,
                help="path to json file that contains dataset information")

ap.add_argument("-d", "--dataset_name",
                required=True,
                type=str,
                help="dataset name")

ap.add_argument("-l", "--segment_length",
                default=3,
                type=float,
                help="length of audio segmentation")

ap.add_argument("-m", "--mode",
                default=1,
                type=int,
                help="mode for segment each audio to one or multiple")

args = vars(ap.parse_args())




def segmentation(dataset_info, filename_generator, segment_length=3, segment_mode=1):
    ## Internal Function for Segmentation
    #
    # inputs: 
    #   dataset_info : The dictionary contains dataset information
    #   filename_generator : generator function that returns the name and label
    #   segment_length : length of segment part (seconds)
    #   segment_mode : (0 : each audio semgment to multi parts, 1 : each audio segment to one part)
    # output :
    #   Files in desired sizes and normalized 


    directory = make_segment_directory(dataset_info, segment_length, segment_mode)

    segmented_directory_format = os.path.join(directory, "{}/{}_{}.wav")
    

    with tqdm(total=None, position=0, leave=True) as pbar:
        for name, label in tqdm(filename_generator , position=0, leave=True):
        #for name, label in filename_generator:
            raw_data, fs = read_wave(name)

            trimed_data = trim_wave(raw_data, segment_length=int(segment_length*fs), segment_mode=segment_mode)
            for counter in range(0, len(trimed_data), int(segment_length*fs)):
                segment_data = trimed_data[counter:counter+int(segment_length*fs)]
                segment_data = normalize(segment_data, segment_length=int(segment_length*fs))

                filename = os.path.basename(name)
                filename = filename.split('.')[0]
                filename = segmented_directory_format.format(label, filename, counter)

                ## For files that are duplicate in name
                if os.path.isfile(filename):
                    filename = os.path.basename(name)
                    filename = filename.split('.')[0]
                    filename = segmented_directory_format.format(label, filename + name.split('/')[-2], counter)

                write_wave(segment_data, filename, fs)
        cleaning_directory_filename(directory)



def segment_dataset(dataset_path, info_path, dataset_name, segment_length=3, segment_mode=1):
    ## Main Function for Segmentation 
    #
    # inputs: 
    #   dataset_path : directory to dataset
    #   info_path : directory to DATASET_INFO.json file
    #   dataset_name : name of desire dataset
    #   segment_length : length of segment part (second)
    # output :
    #   Files in desired sizes and normalized

    dataset_info = read_json(info_path)
    dataset_info[dataset_name]["Directory"] = dataset_path

    if dataset_name == "IEMOCAP":
        dataset_info = dataset_info["IEMOCAP"]
        dataset_generator = iemocap_before_segment(dataset_info)

    elif dataset_name == "EMO-DB":
        dataset_info = dataset_info["EMO-DB"]
        dataset_generator = emodb_before_segment(dataset_info)

    else:
        raise ValueError('Dataset name not Valid!')


    
    segmentation(dataset_info, dataset_generator, segment_length=segment_length, segment_mode=segment_mode)


print("*"*25, args["dataset_name"], "*"*25)
segment_dataset(args["dataset_path"], args["info_path"], args["dataset_name"], args["segment_length"], args["mode"])