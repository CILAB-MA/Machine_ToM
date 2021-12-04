import torch.nn as nn
import torch as tr


class CharNet(nn.Module):
    def __init__(self, num_past, num_input):
        super(CharNet, self).__init__()
        self.conv = nn.Conv2d(num_input, 8, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTMCell(800, 800)
        self.avgpool = nn.AvgPool1d(8)
        self.fc1 = nn.Linear(100, 2)
        self.hidden_size = 800

    def init_hidden(self, batch_size):
        return  (tr.zeros(batch_size, 800, device='cuda'),
                 tr.zeros(batch_size, 800, device='cuda'))

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
        self.conv1 = nn.Conv2d(8, 32, 2, 1)
        self.conv2 = nn.Conv2d(32, 32, 2, 1)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(6)
        self.fc = nn.Linear(32, 5)
        self.device = device
        self.softmax = nn.Softmax()

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
            past_traj, curr_state, target = batch
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
            tot_acc += tr.sum(pred_onehot==targ_onehot).item()
            tot_loss += loss.item()
        return dict(action_acc= tot_acc / 1000, action_loss=tot_loss / (i + 1))

    def evaluate(self, data_loader, is_visualize=False):
        tot_acc = 0
        tot_loss = 0
        for i, batch in enumerate(data_loader):
            with tr.no_grad():

                past_traj, curr_state, target = batch
                past_traj = tr.tensor(past_traj, dtype=tr.float, device=self.device)
                curr_state = tr.tensor(curr_state, dtype=tr.float, device=self.device)
                target = tr.tensor(target, dtype=tr.float, device=self.device)
                criterion = nn.KLDivLoss()
            pred, e_char = self.forward(past_traj, curr_state)
            loss = criterion(pred.log(), target)
            pred_onehot = tr.argmax(pred, dim=-1)
            targ_onehot = tr.argmax(target, dim=-1)
            tot_acc += tr.sum(pred_onehot==targ_onehot).item()
            tot_loss += loss.item()

        dicts = dict()
        if is_visualize:
            dicts['past_traj'] = past_traj[:16].cpu().numpy()
            dicts['curr_state'] = curr_state[:16].cpu().numpy()
            dicts['pred_actions'] = pred[:16].cpu().numpy()
            dicts['e_char'] = e_char.cpu().numpy()
        dicts['action_acc'] = tot_acc / 1000
        dicts['action_loss'] = tot_loss / (i + 1)

        return dicts