import torch
import torch.nn as nn
from torchinfo import summary


class Model(nn.Module):
    def __init__(self, time_slot: int = 64, depth: int = 5, num_class: int = 4, channel: int = 1):
        super(Model, self).__init__()
        self.time_slot = time_slot
        self.depth = depth
        self.num_class = num_class
        self.channel = channel
        self.GRU1 = nn.GRU(input_size=self.channel, hidden_size=time_slot, batch_first=True,
                           num_layers=1, bidirectional=False)
        self.GRU5 = nn.GRU(input_size=time_slot, hidden_size=int(time_slot/2), batch_first=False,
                           num_layers=depth, bidirectional=True)
        self.Dense1000 = nn.Linear(time_slot, 1000)
        self.Dense64 = nn.Linear(1000, time_slot)
        self.DROP = nn.Dropout(0.5)
        self.CLS = nn.Linear(time_slot, num_class)
        self.SOFT = nn.Softmax(dim=2)

    def forward(self, x) -> torch.tensor:
        x,_ = self.GRU1(x)
        x_gru, _ = self.GRU5(x)
        x_den = self.Dense1000(x_gru)
        x_den = self.DROP(x_den)
        x_den = self.Dense64(x_den)
        x_cls = self.CLS(x_den)
        x_return = self.SOFT(x_cls)
        return x_return

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    a = torch.ones([1, 64, 1], ).to('cpu') # data num, timeslot, channel
    model = Model(64, 5, 4, 1) # timeslot, depth, class
    data = model(a)
    print(f"{data.shape}")
    summary(model, size=(32, 64, 1), depth=4)