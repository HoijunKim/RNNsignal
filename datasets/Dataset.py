from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import FileLister, Mapper, Filter, FileOpener, IterDataPipe, CSVDictParser
import numpy as np
import utils


class MultiEMG:
    def __init__(self):
        None

    def row_processer(self, row: list) -> dict:
        """
        :param row:
        :return: label: [gt] data: [0:3]
        """
        label = np.array(row[0], np.int32)  # 64
        labels = np.reshape(np.eye(4)[label.astype(np.int32)], (64, 4))  # time_slot, cls -> one-hot encoding
        return {"label": labels.astype(np.float32), "data": np.array(row[1], dtype=np.float32)}

    def emg_dataset(self, root_dir: str = "./data/train/", window_size: int = 64, step: int = 1) -> Mapper:
        """
        :param root_dir: data location
        :param window_size:
        :param step:
        :return: Mapper(label: [gt] data: [0:3])
        """
        dp = FileLister(root_dir)
        dp = Filter(dp, filter_fn=utils.AddFunc.filter_for_data())
        dp = FileOpener(dp, mode='rt')
        dp = dp.parse_csv(delimiter=",", skip_lines=1)
        dp = dp.rolling(window_size, step)
        return Mapper(dp, self.row_processer)


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




if __name__ == '__main__':
    from torch.utils.data import DataLoader, BatchSampler
    FOLDER = f"./data/train"
    datapipe = MultiEMG.emg_dataset("../data/train", 64, 1)
    print(len(list(enumerate(datapipe))))
    # breakpoint()
    # datapipe2 = EMGDataset("./data/data/train", 64, 1)
    dl = DataLoader(dataset=datapipe, batch_size=32, num_workers=1,)
    print(len(list(enumerate(dl))))
