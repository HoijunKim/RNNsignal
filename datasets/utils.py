import numpy as np


def row_processor(row):
    """
    :param row:
    :return: label: [gt] data: [0:3]
    """
    label = np.array(row[0], np.int32)  # 64
    labels = np.reshape(np.eye(4)[label.astype(np.int32)], (64, 4))  # time_slot, cls -> one-hot encoding
    return {"label": labels.astype(np.float32), "data": np.array(row[1], dtype=np.float32)}


def filter_for_data(filename):
    return filename.endswith(".csv")
