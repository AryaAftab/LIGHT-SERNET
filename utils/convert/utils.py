import os


def list_files(start_path):
    ## this function return list file in directory
    #
    # input :
    #   start_path : input directory
    # output : 
    #   All_filenames : list of file names

    All_filenames = []
    for dirname, _, filenames in os.walk(start_path):
        for filename in filenames:
            path = os.path.join(dirname, filename)
            All_filenames.append(path)
    return All_filenames



def clear_hidden_file(filenames):
    ## this function clear list filename from hidden file
    #
    # inputs:
    #   filenames : filenames contain hidden file
    # output:
    #   new_filenames : cleand filenames

    new_filenames = []
    for filename in filenames:
        splited_filename = filename.split(os.path.sep)

        if splited_filename[-1][0] != ".":
            new_filenames.append(filename)

    return new_filenames