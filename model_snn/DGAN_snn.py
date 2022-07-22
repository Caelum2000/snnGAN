import torch
from torch import nn
from model_snn.nodes import *
from model_snn.encoders import *
import torch.nn.functional as F
import global_v as glv


class Generator_2_MP(nn.Module):

    def __init__(self, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            MPNode()  # nn.Sigmoid()  # nn.Tanh()
        )
        self.sig = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 7, 7)
            x = self.conv1(x)
            x = self.conv2(x)  # (batch_size,1,28,28)
            output.append(x)
        if not self.is_split:
            res_mem = output[-1] / glv.network_config[
                'n_steps']  # (batch_size, 1, 28, 28)
            img = self.sig(res_mem)
            img_spike = img.repeat(glv.network_config['n_steps'], 1, 1, 1, 1)
        else:
            output = torch.stack(output, dim=0)
            img_spike = self.sig(output)
        return img_spike


class Generator_2_MP_weighted_mi_1(nn.Module):

    def __init__(self, weighted_func, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            MPNode()  # nn.Sigmoid()  # nn.Tanh()
        )
        self.sig = nn.Sigmoid()
        self.weighted_fnc = weighted_func

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        weights = []
        infonce_loss = 0
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 7, 7)
            x = self.conv1(x)
            x = self.conv2(x)  # (batch_size,1,28,28)
            x = self.sig(x)
            score, nce_l = self.weighted_fnc(x)
            infonce_loss += nce_l
            weights.append(score)
            output.append(x)
        if not self.is_split:
            weights_tensor = torch.stack(weights, dim=0)  # (n_steps,b,1)
            after_softmax = F.softmax(weights_tensor,
                                      dim=0).unsqueeze(-1).unsqueeze(
                                          -1)  # (n_steps,b,1,1,1)
            output_tensor = torch.stack(output, dim=0)  # (n_steps,b,1,28,28)
            img_spike = torch.sum(output_tensor * after_softmax,
                                  dim=0).unsqueeze(0).repeat(
                                      len(output), 1, 1, 1,
                                      1)  # (n_steps,b,1,28,28)
        else:
            output = torch.stack(output, dim=0)
            img_spike = output
        return img_spike, infonce_loss / len(output)


class Generator_3_MP_Scoring(nn.Module):

    def __init__(self, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),  # 128 * 8 * 8 for 32*32 CIFAR-10
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            LIFNode()  # nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),  # channel: 3
            ScoringMP(scoring_mode=glv.network_config['scoring_mode']
                      )  # nn.Sigmoid()  # nn.Tanh()
        )
        self.sig = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 7, 7)
            x = self.conv1(x)
            x = self.conv2(x)  # (batch_size,1,28,28)
            output.append(x)
        if not self.is_split:
            res_mem = output[-1] / glv.network_config[
                'n_steps']  # (batch_size, 1, 28, 28)
            img = self.sig(res_mem)
            img_spike = img.repeat(glv.network_config['n_steps'], 1, 1, 1, 1)
        else:
            output = torch.stack(output, dim=0)
            img_spike = self.sig(output)
        return img_spike


class Generator_3_MP_Scoring_2(nn.Module):
    '''utilizing sigmoid function before scoring'''

    def __init__(self, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), LIFNode())
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
            ScoringMP(scoring_mode=glv.network_config['scoring_mode']
                      )  # nn.Sigmoid()  # nn.Tanh()
        )
        # self.sig = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        for x in input:
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 7, 7)
            x = self.conv1(x)
            x = self.conv2(x)  # (batch_size,1,28,28), [0,1]
            output.append(x)
        if not self.is_split:
            img = output[-1]
            img_spike = img.repeat(glv.network_config['n_steps'], 1, 1, 1, 1)
        else:
            output = torch.stack(output, dim=0)
            img_spike = output
        return img_spike


class Generator_3_MP_Scoring_2_MI(nn.Module):
    '''utilizing sigmoid function before scoring'''

    def __init__(self, input_dim=100, is_split=False):
        super().__init__()
        self.is_split = is_split
        self.encoder = Encoder(step=glv.network_config['n_steps'],
                               device=glv.network_config['device'],
                               encode_type=glv.network_config['encode_type'])
        self.fc1 = nn.Linear(input_dim, 32 * 32)
        self.br1 = nn.Sequential(
            nn.BatchNorm1d(1024),
            LIFNode()  # nn.ReLU()
        )
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)
        self.br2 = nn.Sequential(
            nn.BatchNorm1d(128 * 7 * 7),
            LIFNode()  # nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), LIFNode())
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
            ScoringMP(scoring_mode=glv.network_config['scoring_mode']
                      )  # nn.Sigmoid()  # nn.Tanh()
        )
        # self.sig = nn.Sigmoid()

    def forward(self, input):
        input = self.encoder(input)
        # input.shape = (num_steps,...)
        output = []
        infonce_loss = 0
        for i, x in enumerate(input):
            x = self.br1(self.fc1(x))
            x = self.br2(self.fc2(x))
            x = x.reshape(-1, 128, 7, 7)
            x = self.conv1(x)
            x, temp = self.conv2(x)  # (batch_size,1,28,28), [0,1]
            # print(temp)
            if i != 0: infonce_loss += temp
            output.append(x)
        if not self.is_split:
            img = output[-1]
            img_spike = img.repeat(glv.network_config['n_steps'], 1, 1, 1, 1)
        else:
            output = torch.stack(output, dim=0)
            img_spike = output
        return img_spike, infonce_loss / (len(input) - 1)
