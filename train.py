from torchmetrics import Accuracy
from datasets.Dataset import emg_dataset
from models.RnnNet import Model
import torch.nn as nn
import torch.distributed
from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchinfo import summary
from utils.AddFunc import check_folder


local_rank = 0
world_size = 0
depth = 4
batch = 32
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
    train = emg_dataset(data_dir="./emg_data/train", window_size=time_slot, channel=channel, step=1)  # 49080
    test = emg_dataset(data_dir='./emg_data/test', window_size=time_slot, channel=channel, step=1)  # 6135
    trainset = DataLoader(dataset=train, batch_size=batch, shuffle=shuffle, num_workers=num_worker,
                          pin_memory=pin, prefetch_factor=prefetch, persistent_workers=persistent)
    nb = len(list(enumerate(trainset)))
    # hl = list(enumerate(trainset))
    # print(hl[0])
    # breakpoint()
    model = Model(time_slot, depth, num_classes, channel).to(device)

    if opt == f'SGD':
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentums, weight_decay=weight_dc)
    elif opt == f'Adam':
        optimizer = Adam(model.parameters(), lr=learning_rate)
    elif opt == f'RMSProp':
        optimizer = RMSprop(model.parameters(), lr=learning_rate)
    else:
        print(f"Opt error !!")
        breakpoint()

    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.MSELoss()
    summary(model,size=(batch,time_slot,4),)
    for epoch in tqdm(range(start_epoch, end_epochs), desc=f'Epoch', disable=False):
        model.train()
        print(f'{"Gpu_mem":10s} {"total":>10s} ')
        pbar = tqdm(enumerate(trainset), total=nb, desc=f'batch', leave=True, disable=False)
        for batch_idx, (features) in pbar:
            targets, data = features["label"].to(device), features["data"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                output = model(data)
                losses = criterion(output, targets)
                # acces = accuracy(output, targets)
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
            mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = f'{mem:10s} {losses.mean():10.6g}'
            pbar.set_description(s)

    torch.save(model.state_dict(), f'result2'+'.pt')
