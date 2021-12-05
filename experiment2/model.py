import torch.nn as nn
import torch as tr
import torch.nn.functional as F


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

class softmax_SR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        sr = F.softmax(x.reshape(x.size(0), x.size(1), -1), dim=2)
        return sr

class CharNet(nn.Module):
    def __init__(self, num_past, num_input, num_exp=1):
        super(CharNet, self).__init__()
        self.num_exp = num_exp
        self.conv1 = ResNetBlock(num_input, 4, 1)
        self.conv2 = ResNetBlock(4, 8, 1)
        self.conv3 = ResNetBlock(8, 16, 1)
        self.conv4 = ResNetBlock(16, 32, 1)
        self.conv5 = ResNetBlock(32, 32, 1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.lstm = nn.LSTMCell(32, 64)
        self.avgpool = nn.AvgPool2d(11)
        self.fc64_2 = nn.Linear(64, 2)
        self.fc64_8 = nn.Linear(64, 8)
        self.fc32_2 = nn.Linear(32, 2)
        self.fc32_8 = nn.Linear(32, 8)
        self.hidden_size = 64

    def init_hidden(self, batch_size):
        return (tr.zeros(batch_size, 64, device='cuda'),
                tr.zeros(batch_size, 64, device='cuda'))

    def forward(self, obs):
        # batch, num_past, num_step, channel, height, width
        b, num_past, num_step, c, h, w = obs.shape
        past_e_char = []
        for p in range(num_past):
            prev_h = self.init_hidden(b)
            obs_past = obs[:, p] #batch(0), num_step(1), channel(2), height(3), width(4)
            obs_past = obs_past.permute(1, 0, 2, 3, 4)
            obs_past = obs_past.reshape(-1, c, h, w)

            x = self.conv1(obs_past)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.conv5(x)
            x = self.relu(x)
            x = self.bn(x)
            x = self.avgpool(x)

            if self.num_exp == 2:
                x = x.view(num_step, b, -1)
                x = x.transpose(1, 0)
                x = self.fc32_2(x)  ## batch, output
            else:
                outs = []
                for step in range(num_step):
                    next_h, next_c = self.lstm(x.view(num_step, b, -1)[step], prev_h)
                    outs.append(next_h)
                x = tr.stack(outs, dim=0) ## step, batch, output
                x = x.transpose(1, 0) ## batch, step, output
                x = self.fc64_2(x) ## batch, output

            # sum the each step e_char
            e_char = []
            for i in range(len(x)):
                e_char.append(sum(x[i]))
            e_char_sum = tr.stack(e_char, dim=0)
            final_e_char = e_char_sum

            if self.num_exp == 2 or self.num_exp == 3:
                ## stack_num_past
                past_e_char.append(e_char_sum)

                ## sum_num_past
                past_e_char = tr.stack(past_e_char, dim=0)
                past_e_char = past_e_char.transpose(1, 0)

                past_e_char_sum = []
                for i in range(len(past_e_char)):
                    past_e_char_sum.append(sum(past_e_char[i]))

                past_e_char_sum = tr.stack(past_e_char_sum, dim=0)
                final_e_char = past_e_char_sum

        return final_e_char

class PredNet(nn.Module):
    def __init__(self, num_past, num_input, device):
        super(PredNet, self).__init__()

        self.e_char = CharNet(num_past, num_input, num_exp=2)
        self.conv1 = ResNetBlock(8, 8, 1)
        self.conv2 = ResNetBlock(8, 16, 1)
        self.conv3 = ResNetBlock(16, 16, 1)
        self.conv4 = ResNetBlock(16, 32, 1)
        self.conv5 = ResNetBlock(32, 32, 1)
        self.avgpool = nn.AvgPool2d(11)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.action_fc = nn.Linear(32, 5)
        self.comsumption_fc = nn.Linear(32, 4)
        self.device = device
        self.softmax = nn.Softmax()

        self.action_head = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(11),
            nn.Flatten(),
            nn.Linear(32,5),
            nn.LogSoftmax()
        )

        self.consumption_head = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(11),
            nn.Flatten(),
            nn.Linear(32,4),
            nn.Sigmoid()
        )

        self.representation_head = nn.Sequential(
            nn.Conv2d(32, 32, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1, 1),
            softmax_SR() # each channel is for gamma=0.5, 0.9, 0.99
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

        x = self.conv1(x_concat)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.bn(x)

        action = self.action_head(x)
        consumption = self.consumption_head(x)
        sr = self.representation_head(x)

        return action, consumption, sr

    def train(self, data_loader, optim):
        tot_loss = 0
        a_loss = 0
        c_loss = 0
        s_loss = 0
        action_acc = 0
        consumption_acc = 0

        criterion_nll = nn.NLLLoss()
        criterion_bce = nn.BCELoss()
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        for i, batch in enumerate(data_loader):
            past_traj, curr_state, target_action, target_consume, target_sr, target_v, dones = batch
            past_traj = past_traj.float().cuda()
            curr_state = curr_state.float().cuda()
            target_action = target_action.long().cuda().squeeze(-1)
            target_consume_onehot = target_consume.float().cuda()
            target_sr = target_sr.float().cuda()
            target_v = target_v.float().cuda()


            pred_action, pred_consumption, pred_sr = self.forward(past_traj, curr_state)
            action_loss = criterion_nll(pred_action, target_action)
            consumption_loss = criterion_bce(pred_consumption, target_consume_onehot)
            sr_loss = criterion_kl(pred_sr.log().transpose(1, 2), target_sr.flatten(1, 2))

            optim.zero_grad()
            loss = action_loss + consumption_loss + sr_loss
            loss.mean().backward()
            optim.step()
            pred_action_ind = tr.argmax(pred_action, dim=-1)
            pred_consumption_ind = tr.argmax(pred_consumption, dim=-1)
            targ_consumption_ind = tr.argmax(target_consume_onehot, dim=-1)

            a_loss += action_loss.item()
            c_loss += consumption_loss.item()
            s_loss += sr_loss.item()
            tot_loss += loss.item()

            action_acc += tr.sum(pred_action_ind == target_action).item()
            consumption_acc += tr.sum(pred_consumption_ind == targ_consumption_ind).item()

        dicts = dict(action_acc=action_acc / 1000, consumption_acc=consumption_acc / 1000,
                     action_loss=a_loss / (i + 1), consumption_loss=c_loss / (i + 1),
                     sr_loss=sr_loss / (i + 1), total_loss=tot_loss / (i + 1))
        return dicts

    def evaluate(self, data_loader):

        tot_loss = 0
        a_loss = 0
        c_loss = 0
        s_loss = 0
        action_acc = 0
        consumption_acc = 0

        criterion_nll = nn.NLLLoss()
        criterion_bce = nn.BCELoss()  # reduction sum or mean?
        criterion_kl = nn.KLDivLoss(reduction='batchmean')

        for i, batch in enumerate(data_loader):
            past_traj, curr_state, target_action, target_consume, target_sr, target_v, dones = batch
            past_traj = past_traj.float().cuda()
            curr_state = curr_state.float().cuda()
            target_action = target_action.long().cuda().squeeze(-1)
            target_consume_onehot = target_consume.float().cuda()
            target_sr = target_sr.float().cuda()
            target_v = target_v.float().cuda()
            # y_onehot = tr.zeros(target_consume.shape[0], 4).cuda()
            # target_consume_onehot = y_onehot.scatter_(1, target_consume, 1)

            pred_action, pred_consumption, pred_sr = self.forward(past_traj, curr_state)
            action_loss = criterion_nll(pred_action, target_action)
            consumption_loss = criterion_bce(pred_consumption, target_consume_onehot)
            # consumption_loss = criterion_nll(pred_consumption, target_consume)
            sr_loss = criterion_kl(pred_sr.log().transpose(1, 2), target_sr.flatten(1, 2))


            loss = action_loss + consumption_loss + sr_loss


            pred_action_ind = tr.argmax(pred_action, dim=-1)
            pred_consumption_ind = tr.argmax(pred_consumption, dim=-1)
            targ_consumption_ind = tr.argmax(target_consume_onehot, dim=-1)

            a_loss += action_loss.item()
            c_loss += consumption_loss.item()
            s_loss += sr_loss.item()
            tot_loss += loss.item()

            action_acc += tr.sum(pred_action_ind == target_action).item()
            consumption_acc += tr.sum(pred_consumption_ind == targ_consumption_ind).item()

        act_preds = pred_action[:16].detach().cpu().numpy()
        sr_preds = pred_sr[:16].reshape(-1, 3, 11, 11).detach().cpu().numpy()
        con_preds = pred_consumption[:16].detach().cpu().numpy()

        dicts = dict(action_acc=action_acc / 1000, consumption_acc=consumption_acc / 1000,
                     action_loss=a_loss / (i + 1), consumption_loss=c_loss / (i + 1),
                     sr_loss=sr_loss / (i + 1), total_loss=tot_loss / (i + 1))
        return dicts
