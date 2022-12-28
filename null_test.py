import pandas as pd
import numpy as np
import os

dir_path = './class3/train/'
dir_list = os.listdir(dir_path)
sum_zero = 0
for i, j in enumerate(dir_list):
    data = pd.read_csv(os.path.join(dir_path, j))
    data_cls = np.array(data)[:, 3]
    sum_zero += (len(data_cls) - np.count_nonzero(data_cls))/len(data_cls)
print(sum_zero/len(dir_list))