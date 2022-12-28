import torch
import torchdata.datapipes.iter
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import FileLister, Mapper, Filter, FileOpener, IterDataPipe, CSVDictParser
import numpy as np
# mps_device = torch.device("mps")
import os

count = 0
def filter_for_data(filename):
    return filename.endswith(".csv")


@functional_datapipe("rolling")
class RollingWindow(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, window_size, step=1) -> None:
        super().__init__()
        self.source_dp = source_dp
        self.window_size = window_size
        self.step = step

    def __iter__(self):
        it = iter(self.source_dp)
        cur = []
        cur2 = []
        while True:
            try:
                while len(cur) < self.window_size:
                    a = next(it)
                    cur.append(a[-1])
                    cur2.append(a[0:3])
                yield np.array(cur), np.array(cur2)  # torch.tensor ?
                for _ in range(self.step):
                    if cur:
                        cur.pop(0)
                        cur2.pop(0)
                    else:
                        next(it)
            except StopIteration:
                return


def row_processer(row):
    label = np.array(row[0], np.int32)  # 64
    labels = np.reshape(np.eye(4)[label.astype(np.int32)],(64, 4)) # time_slot, cls -> one-hot encoding
    return {"label": labels.astype(np.float32), "data":np.array(row[1], dtype = np.float32)}


def emg_dataset(root_dir="./class3/train/",window_size:int = 64, step:int = 1):
    """
    :param root_dir: data location
    :return: label: [gt] data: [0:3]
    """

    dp = FileLister(root_dir)
    dp = Filter(dp, filter_fn=filter_for_data)
    dp = FileOpener(dp, mode='rt')
    dp = dp.parse_csv(delimiter=",", skip_lines=1)
    dp = dp.rolling(window_size, step)
    return Mapper(dp, row_processer)


# class EMGDataset(torchdata.datapipes.iter.IterDataPipe):
#     def __init__(self, root:str = "./data/class3/train", window_size:int = 64, step:int = 1):
#         super(EMGDataset, self).__init__()
#         self.root = root
#         self.window_size = window_size
#         self.step = step
#
#     def __iter__(self):
#         dp = FileLister(self.root)
#         dp = Filter(dp, filter_fn = filter_for_data)
#         dp = FileOpener(dp, mode = 'rt')
#         dp = dp.parse_csv(delimiter = ",", skip_lines = 1)
#         dp = dp.rolling(self.window_size, self.step)
#         dp = dp.map(row_processer)
#         return dp
#
#     def __len__(self):
#         return {"train": 49080, "test": 6135}



if __name__ == '__main__':
    from torch.utils.data import DataLoader, BatchSampler
    FOLDER = f"./class3/train"
    datapipe = emg_dataset("./class3/train", 64, 1)

    for i, data in enumerate(datapipe):
        c = i
    print(c)
    # breakpoint()
    # datapipe2 = EMGDataset("./data/class3/train", 64, 1)
    dl = DataLoader(dataset=datapipe, batch_size=32, num_workers=1,)
    for i, data in enumerate(dl):
        # print(f'Labels batch shape: {data["label"]}{data["label"].size()}\n')
        # print(f'Feature batch shape:{data["data"]}{data["data"].size()}\n')
        count = i

    print(count)
