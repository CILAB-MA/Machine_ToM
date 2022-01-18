import torch.nn as nn
import torch as tr

class A3CNetwork(nn.Module):

    def __init__(self, num_input, num_action):
        super(A3CNetwork, self).__init__()
        self.conv1 = nn.Conv2d(num_input, 16, 8, 4)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_fc = nn.Linear(500, 256) # 500 will be changed to conv flatten
        self.lstm = nn.LSTM(256 + num_action + 1, 256) # +1 for reward

        self.pi = nn.Sequential(
            nn.Linear(256, num_action),
            nn.Softmax()
        )
        self.v = nn.Linear(256, 1)


    def forward(self, obs, action_reward):
        # obs = (b, s, h, w, c), action = (b, s, a_r_h)
        b, s, h, w, c = obs.shape
        obs = obs.permute(1, 0, 4, 2, 3)
        obs = obs.reshape(-1, c, h, w)
        x = self.relu(self.conv1(obs))
        x = self.relu(self.conv2(x))

        x = x.view(s, b, 500) # 500 will be changed to conv flatten dimension
        action_reward = action_reward.permute(1, 0, 2) # (s, b, a_r_h)

        #x = self.relu(self.conv_fc(x))
        x = tr.concat([x, action_reward], dim=-1)
        x = self.lstm(x)
        x = x.transpose(0, 1).reshape(b, -1)
        pi = self.pi(x)
        v = self.v(x)
        return pi, v



