from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import FileLister, Mapper, Filter, FileOpener, IterDataPipe
import numpy as np
from utils import row_processor, filter_for_data

def emg_dataset(data_dir: str = "./data/train/", window_size: int = 64, step: int = 1):
    """
    :param data_dir: data location
    :param window_size:
    :param step:
    :return: Mapper(label: [gt] data: [0:3])
    """
    dp = FileLister(data_dir)
    dp = Filter(dp, filter_fn=filter_for_data)
    dp = FileOpener(dp, mode='rt')
    dp = dp.parse_csv(delimiter=",", skip_lines=1)
    dp = dp.rolling(window_size, step)
    return Mapper(dp, row_processor)


@functional_datapipe("rolling")
class RollingWindow(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe, window_size: int = 64, step: int = 1):
        super().__init__()
        self.source_dp = source_dp
        self.window_size = window_size
        self.step = step

    def __iter__(self):
        it = iter(self.source_dp)
        label, data = [], []
        while True:
            try:
                while len(label) < self.window_size:
                    a = next(it)
                    label.append(a[-1])
                    data.append(a[0:3])
                yield np.array(label), np.array(data)  # torch.tensor ?
                for _ in range(self.step):
                    if label:
                        label.pop(0)
                        data.pop(0)
                    else:
                        next(it)
            except StopIteration:
                return


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    FOLDER = f"./data/train"
    datapipe = emg_dataset("../data/train", 64, 1)
    print(len(list(enumerate(datapipe))))
    dl = DataLoader(dataset=datapipe, batch_size=32, num_workers=1,)
    print(len(list(enumerate(dl))))
