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


class ClickNet(nn.Module):

    def __init__(self, n_features, n_hidden, n_sequence, n_layers, n_classes):
        super(ClickNet, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_sequence = n_sequence
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=n_hidden, num_layers=n_layers, batch_first=True)
        self.linear_1 = nn.Linear(in_features=n_hidden, out_features=128)
        self.dropout_1 = nn.Dropout(p=0.2)
        self.linear_2 = nn.Linear(in_features=128, out_features=n_classes)

    def forward(self, x) -> torch.Tensor:
        print(x.size())
        self.hidden = (
            torch.zeros(self.n_layers, x.shape[0], self.n_hidden),
            torch.zeros(self.n_layers, x.shape[0], self.n_hidden)
        )

        out, (hs, cs) = self.lstm(x.view(len(x), self.n_sequence, -1), self.hidden)
        out = out[:, -1, :]
        out = self.linear_1(out)
        out = self.dropout_1(out)
        out = self.linear_2(out)

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = torch.ones([1,1,64,3], ).to(device)
    model = Model(64, 4, 3)
    data = model(a)
    print(f"{data.shape}")
    summary(model, size=(32, 64, 3), depth = 4)