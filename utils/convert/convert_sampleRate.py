import os
import argparse
from tqdm import tqdm
import shutil

from utils import *


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--start_path",
                required=True,
                type=str,
                help="path to input files")

ap.add_argument("-o", "--convert_folder",
                required=True,
                type=str,
                help="folder name for converted files")

ap.add_argument("-s", "--sample_rate",
                default=16000,
                type=int,
                help="sample rate for converted files")

args = vars(ap.parse_args())




def samplerate_converter(start_path ,convert_folder, sample_rate=16000):
    ## this function convert audio to another sample rate(default 16000Hz)
    #
    # inputs:
    #   start_path : input directory
    #   convert_folder : output folder
    #	sample_rate : target sample rate
    # output:
    #   converted file

    os.chdir(start_path)
    All_filenames = list_files('.')
    All_filenames = clear_hidden_file(All_filenames)

    with tqdm(total=len(All_filenames), position=0, leave=True) as pbar:
        for filename in tqdm(All_filenames , position=0, leave=True):
            name = os.path.dirname(filename)
            new_name = os.path.join(convert_folder, name[2:])
            os.makedirs(new_name, exist_ok=True)

            dst = os.path.join(convert_folder, filename[2:])
            if filename.endswith(".wav"):
                os.system('ffmpeg -hide_banner -loglevel error -i {} -acodec pcm_s16le -ac 1 -ar {} {}'.format(filename, sample_rate, dst))
            else:
                shutil.copyfile(filename, dst)

    shutil.move(convert_folder, "..", copy_function=shutil.copytree) 



output_path = args['start_path'].split(os.path.sep)
output_path[-1] = args['convert_folder']
output_path = "/".join(output_path)

print(f"Start Path : {args['start_path']}\nOutput Path : {output_path}\nOutput Sample Rate : {args['sample_rate']}")
samplerate_converter(args["start_path"], args["convert_folder"], args["sample_rate"])