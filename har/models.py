"""Backbones expressed as an ordered list of 'layer groups' so freezing / per-layer LR / discrepancy are uniform.
Every group maps (B,C,T) -> (B,C',T); the head is GAP + Linear."""
import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, groups, names, feat_dim, n_classes):
        super().__init__()
        self.groups = nn.ModuleList(groups)
        self.names = list(names)
        self.head = nn.Linear(feat_dim, n_classes)

    def forward(self, x, return_feats=False):
        feats = []
        for g in self.groups:
            x = g(x)
            if return_feats:
                feats.append(x.mean(-1))          # global-average-pooled representation of this layer
        out = self.head(x.mean(-1))
        return (out, feats) if return_feats else out


def conv_bn(i, o, k):
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm1d(o), nn.ReLU())


class ResBlock(nn.Module):
    def __init__(self, i, o, pool):
        super().__init__()
        self.c1 = conv_bn(i, o, 5)
        self.c2 = nn.Sequential(nn.Conv1d(o, o, 5, padding=2, bias=False), nn.BatchNorm1d(o))
        self.skip = nn.Identity() if i == o else nn.Sequential(nn.Conv1d(i, o, 1, bias=False), nn.BatchNorm1d(o))
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()

    def forward(self, x):
        return self.pool(torch.relu(self.c2(self.c1(x)) + self.skip(x)))


def resnet1d(n_classes=5, in_ch=3):
    g = [conv_bn(in_ch, 32, 7), ResBlock(32, 32, False), ResBlock(32, 64, True), ResBlock(64, 128, True)]
    return Net(g, ["stem", "res1", "res2", "res3"], 128, n_classes)


class LSTMGroup(nn.Module):
    def __init__(self, i, h):
        super().__init__()
        self.lstm = nn.LSTM(i, h, num_layers=2, batch_first=True, dropout=0.2)

    def forward(self, x):
        return self.lstm(x.transpose(1, 2))[0].transpose(1, 2)


def deepconvlstm(n_classes=5, in_ch=3):
    """DeepConvLSTM (Ordonez & Roggen 2016): 4 conv layers + 2 LSTM layers."""
    g = [nn.Sequential(nn.Conv1d(in_ch if i == 0 else 64, 64, 5, padding=2), nn.ReLU()) for i in range(4)]
    g.append(LSTMGroup(64, 128))
    return Net(g, ["conv1", "conv2", "conv3", "conv4", "lstm"], 128, n_classes)


class Inception(nn.Module):
    def __init__(self, i, nf=16, ks=(5, 11, 21)):
        super().__init__()
        self.bott = nn.Conv1d(i, nf, 1, bias=False) if i > 1 else nn.Identity()
        b = nf if i > 1 else i
        self.convs = nn.ModuleList([nn.Conv1d(b, nf, k, padding=k // 2, bias=False) for k in ks])
        self.mp = nn.Sequential(nn.MaxPool1d(3, 1, 1), nn.Conv1d(i, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * (len(ks) + 1))

    def forward(self, x):
        z = self.bott(x)
        return torch.relu(self.bn(torch.cat([c(z) for c in self.convs] + [self.mp(x)], 1)))


def inceptiontime(n_classes=5, in_ch=3):
    """InceptionTime-lite (Fawaz et al. 2020), depth 3, no residual shortcut."""
    g = [Inception(in_ch), Inception(64), Inception(64)]
    return Net(g, ["inc1", "inc2", "inc3"], 64, n_classes)


ARCHS = {"resnet1d": resnet1d, "deepconvlstm": deepconvlstm, "inceptiontime": inceptiontime}


def build_model(arch="resnet1d", n_classes=5):
    return ARCHS[arch](n_classes)


def count_params(model, trainable_only=True):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
