from torchmetrics import Accuracy
from datasets.Dataset import emg_dataset
from models.RnnNet import Model
import torch.distributed
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
from utils.AddFunc import check_folder
import time

local_rank = 0
world_size = 0
depth = 4
batch = 1
time_slot = 64
channel = 1
shuffle = True
num_worker = 1
pin = True
prefetch = 2
persistent = True
opt = f'SGD'
learning_rate = 0.001
amp = False
momentums = 0.9
weight_dc = 0.00005
num_classes = 4

start_epoch = 0
end_epochs = 300
schdual = 'None'
loss_list = []
accdet_list = []
acccls_list = []
result = './timestep_' + str(time_slot) + '/'
check_folder(result)


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda', local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device(f'mps', local_rank)
    else:
        device = torch.device(f'cpu')
    print(device)
    accuracy = Accuracy(task="multiclass", num_classes=4)
    # 2. dataloader
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test = emg_dataset(data_dir='./emg_data/test', window_size=time_slot, channel=channel, step=64)  # 6135
    testset = DataLoader(dataset=test, batch_size=batch, num_workers=num_worker,
                          pin_memory=pin, prefetch_factor=prefetch, persistent_workers=persistent)
    nb = len(list(enumerate(testset)))

    model = Model(batch, depth, num_classes).to(device)

    if opt == f'SGD':
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentums, weight_decay=weight_dc)
    elif opt == f'Adam':
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif opt == f'RMSProp':
        optimizer = RMSprop(model.parameters(), lr=learning_rate)
    else:
        print(f"Opt error !!")
        breakpoint()

    summary(model,size=(batch,time_slot,4),)
    print(f'{"Gpu_mem":10s} {"total":>10s} ')
    pbar = tqdm(enumerate(testset), total=nb, desc=f'batch', leave=True, disable=False)
    for batch_idx, (features) in pbar:
        st = time.time()
        targets, data = features["label"].to(device), features["data"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            output = model(data)
            cls_out = torch.argmax(output, dim=2)
            # acces = accuracy(output, targets)
        mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
        s = f'{mem:10s}'
        pbar.set_description(s)
        # print(output[0])
        # print(cls_out[0])
        # print(cls_out.shape)
        print(time.time() - st)
