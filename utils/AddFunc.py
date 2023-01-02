import os


def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def filter_for_data(filename):
    return filename.endswith(".csv")
