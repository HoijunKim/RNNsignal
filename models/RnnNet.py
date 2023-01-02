import torch
import torch.nn as nn
from torchinfo import summary


class Model(nn.Module):
    def __init__(self, time_slot: int = 64, depth: int = 5, num_class: int = 4):
        super(Model, self).__init__()
        self.time_slot = time_slot
        self.depth = depth
        self.num_class = num_class
        self.GRU1 = nn.GRU(input_size=3, hidden_size=int(time_slot / 2), batch_first=True,
                           num_layers=1, bidirectional=True)
        self.GRU5 = nn.GRU(input_size=32, hidden_size=int(time_slot / 2), batch_first=True,
                           num_layers=depth, bidirectional=True)
        self.CLS = nn.Linear(time_slot, num_class)
        self.SOFT = nn.Softmax(dim=2)

    def forward(self, x) -> torch.tensor:
        x,h = self.GRU1(x)
        x_gru, _ = self.GRU5(x)
        x_cls = self.CLS(x_gru)
        x_return = self.SOFT(x_cls)
        return x_return

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = torch.ones([1, 1, 64, 3], ).to(device)
    model = Model(64, 4, 3)
    data = model(a)
    print(f"{data.shape}")
    summary(model, size=(32, 64, 3), depth=4)