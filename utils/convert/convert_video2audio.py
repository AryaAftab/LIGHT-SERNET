import os
from tqdm import tqdm
import argparse
import shutil

from utils import *



# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_directory",
                required=True,
                type=str,
                help="path for video files")

ap.add_argument("-o", "--output_folder",
                required=True,
                type=str,
                help="path for audio files")

ap.add_argument("-s", "--sample_rate",
                default=16000,
                type=int,
                help="sample rate for converted audio files")

args = vars(ap.parse_args())





def video2audio_converter(input_directory, output_folder, sample_rate=16000):
    ## this function extract audio(.wav) from video(.mp4)
    #
    # inputs:
    #   input_directory : input directory
    #   output_folder : output directory
    #   sample_rate : target sample rate
    # output:
    #   converted file
    
    os.chdir(input_directory)
    filenames = list_files('.')
    filenames = clear_hidden_file(filenames)

    with tqdm(total=len(filenames), position=0, leave=True) as pbar:
        for filename in tqdm(filenames , position=0, leave=True):
            name = os.path.dirname(filename)
            new_name = os.path.join(output_folder, name[2:])
            os.makedirs(new_name, exist_ok=True)

            dst = os.path.join(output_folder, filename[2:])
            if(filename.endswith(".mp4")):
                actual_filename = filename[:-4]
                os.system('ffmpeg -hide_banner -loglevel error -i {} -acodec pcm_s16le -ac 1 -ar {} {}/{}.wav'.\
                                    format(filename, sample_rate, output_folder, actual_filename))
            else:
                shutil.copyfile(filename, dst)

    shutil.move(output_folder, "..", copy_function=shutil.copytree) 


output_path = args['input_directory'].split(os.path.sep)
output_path[-1] = args['output_folder']
output_path = "/".join(output_path)

print(f"Video Path : {args['input_directory']}\nAudio Path : {output_path}\nAudio Sample Rate : {args['sample_rate']}")
video2audio_converter(args["input_directory"], args["output_folder"], args["sample_rate"])