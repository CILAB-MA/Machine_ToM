import torch.nn as nn
import torch as tr

class ResNetBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * ResNetBlock.expansion, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(out_channels * ResNetBlock.expansion),
        )

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU()

        if stride != 1 or in_channels != ResNetBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * ResNetBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResNetBlock.expansion)
            )

    def forward(self, x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x



class CharNet(nn.Module):
    def __init__(self, num_past, num_input):
        super(CharNet, self).__init__()
        self.conv1 = ResNetBlock(num_input, 4, 2)
        self.conv2 = ResNetBlock(4, 8, 2)
        self.conv3 = ResNetBlock(8, 16, 2)
        self.conv4 = ResNetBlock(16, 32, 2)
        self.conv5 = ResNetBlock(32, 32, 2)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTMCell(32, 64)
        self.avgpool = nn.AvgPool1d(8)
        self.fc1 = nn.Linear(8, 2)
        self.hidden_size = 64

    def init_hidden(self, batch_size):
        return (tr.zeros(batch_size, 64, device='cuda'),
                tr.zeros(batch_size, 64, device='cuda'))

    def forward(self, obs):
        # batch, num_past, step, channel , height, width
        b, num_past, num_step, c, h, w = obs.shape
        e_char_sum = 0
        for p in range(num_past):
            prev_h = self.init_hidden(b)
            obs_past = obs[:, p]
            obs_past = obs_past.permute(1, 0, 2, 3, 4)

            obs_past = obs_past.reshape(-1, c, h, w)
            x = self.conv(obs_past)
            x = self.relu(x)
            outs = []
            for step in range(num_step):
                out, prev_h = self.lstm(x.view(num_step, b, -1)[step], prev_h)
                outs.append(out)
            x = tr.stack(outs, dim=0)
            x = x.transpose(1, 0)
            x = self.avgpool(x)
            x = x.squeeze(1)
            x = self.fc1(x)
            e_char_sum += x

        return e_char_sum

class PredNet(nn.Module):
    def __init__(self, num_past, num_input, device):
        super(PredNet, self).__init__()
        self.e_char = CharNet(num_past, num_input)
        self.conv1 = ResNetBlock(8, 8, 2)
        self.conv2 = ResNetBlock(8, 16, 2)
        self.conv3 = ResNetBlock(16, 16, 2)
        self.conv4 = ResNetBlock(16, 32, 2)
        self.conv5 = ResNetBlock(32, 32, 2)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(6)
        self.fc = nn.Linear(32, 5)
        self.device = device
        self.softmax = nn.Softmax()

        self.action_head = nn.Sequential(
            nn.Conv2d(32, 32, 2, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(32),
            nn.Linear(32, 5),
            nn.Softmax()
        )

        self.consumption_head = nn.Sequential(
            nn.Conv2d(32, 32, 2, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(32),
            nn.Linear(32, 4),
            nn.Sigmoid(),
        )

        self.representation_head = nn.Sequential(
            nn.Conv2d(32, 32, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 2, 1),
            nn.Softmax(dim=1), # each channel is for gamma=0.5, 0.9, 0.99
        )
    def init_hidden(self, batch_size):
        return self.e_char.init_hidden(batch_size)

    def forward(self, past_traj, obs):
        b, h, w, c = obs.shape
        obs = obs.permute(0, 3, 1, 2)
        _, _, s, _, _, _ = past_traj.shape
        if s == 0:
            e_char = tr.zeros((b, 2, h, w), device=self.device)
        else:
            e_char_2d = self.e_char(past_traj)
            e_char = e_char_2d.unsqueeze(-1).unsqueeze(-1)
            e_char = e_char.repeat(1, 1, h, w)
        x_concat = tr.cat([e_char, obs], axis=1)

        x = self.relu(self.conv1(x_concat))
        x = self.relu(self.conv2(x))
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)

        out = self.softmax(self.fc(x))

        return out, e_char_2d

    def train(self, data_loader, optim):
        tot_acc = 0
        tot_loss = 0
        for i, batch in enumerate(data_loader):
            past_traj, curr_state, target, ind = batch
            past_traj = tr.tensor(past_traj, dtype=tr.float, device=self.device)
            curr_state = tr.tensor(curr_state, dtype=tr.float, device=self.device)
            target = tr.tensor(target, dtype=tr.float, device=self.device)
            criterion = nn.KLDivLoss()

            pred, _ = self.forward(past_traj, curr_state)
            loss = criterion(pred.log(), target)
            loss.backward()
            optim.step()
            pred_onehot = tr.argmax(pred, dim=-1)
            targ_onehot = tr.argmax(target, dim=-1)
            tot_acc += tr.sum(pred_onehot == targ_onehot).item()
            tot_loss += loss.item()
        return tot_acc / 1000, tot_loss / (i + 1)

    def evaluate(self, data_loader):
        tot_acc = 0
        tot_loss = 0
        for i, batch in enumerate(data_loader):
            with tr.no_grad():
                past_traj, curr_state, target, ind = batch
                past_traj = tr.tensor(past_traj, dtype=tr.float, device=self.device)
                curr_state = tr.tensor(curr_state, dtype=tr.float, device=self.device)
                target = tr.tensor(target, dtype=tr.float, device=self.device)
                criterion = nn.KLDivLoss()
            pred, e_char = self.forward(past_traj, curr_state)
            loss = criterion(pred.log(), target)
            pred_onehot = tr.argmax(pred, dim=-1)
            targ_onehot = tr.argmax(target, dim=-1)
            tot_acc += tr.sum(pred_onehot == targ_onehot).item()
            tot_loss += loss.item()


