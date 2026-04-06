import os


def check_output_folder(path):
    if os.path.isdir(path):
        pass
    else:
        os.makedirs(path)
        print(path + " folder created")
