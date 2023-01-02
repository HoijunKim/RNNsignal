import torch
import torch.nn as nn
from torchinfo import summary


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
    model = ClickNet(64, 4, 3)
    data = model(a)
    print(f"{data.shape}")
    summary(model, size=(32, 64, 3), depth = 4)