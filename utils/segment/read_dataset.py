import os
import re




def iemocap_before_segment(dataset_info):
    ## This is a generator function that returns the name and label
    ## of files from the IEMOCAP database before segmentation.
    #
    # input :
    #   dataset_info : The dictionary contains dataset information

    pattern = '{}\t(.*?)\t\['

    classes = dataset_info["Class"]
    dataset_directory = dataset_info["Directory"]

    label_format = 'Session{}/dialog/EmoEvaluation/{}.txt'
    label_format = os.path.join(dataset_directory, label_format)


    for dirname, _, filenames in os.walk(dataset_directory):
        for filename in filenames:
            if filename[0] == ".": # for scape hidden file
                continue

            if not filename.endswith(".wav"): # for scape None-wav file
                continue


            wavename = filename[:-4]
            label = label_format.format(wavename[4], wavename[:-5])

            filename = os.path.join(dirname, filename)

            with open(label, 'r') as text:
                content = text.read()
                label = re.findall(pattern.format(wavename), content)[0]
            
            yield (filename, classes[label])



def emodb_before_segment(dataset_info):
    ## This is a generator function that returns the name and label
    ## of files from the EMO-DB database before segmentation.
    #
    # input :
    #   dataset_info : The dictionary contains dataset information

    dataset_directory = dataset_info["Directory"]

    classes = dataset_info["Class"]
    for dirname, _, filenames in os.walk(dataset_directory):
        for filename in filenames:
            if filename[0] == ".": # for scape hidden file
                continue

            if not filename.endswith(".wav"): # for scape None-wav file
                continue

            label = filename[-6]

            filename = os.path.join(dirname, filename)
            
            yield (filename, classes[label])