from torch.utils.data import Dataset
import numpy as np

class ToMDataset(Dataset):

    def __init__(self, episodes, curr_state, target_action, target_prefer=None,
                 target_sr=None, target_v=None, dones=None, exp='exp1'):
        self.episodes = episodes
        self.curr_state = curr_state

        self.target_action = target_action
        self.target_prefer = target_prefer
        self.target_sr = target_sr
        self.target_v = target_v

        self.dones = dones
        self.exp = exp

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, ind):
        if self.exp == 'exp1':
            return self.episodes[ind], self.curr_state[ind], self.target_action[ind]
        else:
            return self.episodes[ind], self.curr_state[ind], self.target_action[ind],\
                   self.target_prefer[ind], self.target_sr[ind], self.target_v[ind], self.dones[ind]


def save_data(npy_data, number, alpha, fn=None):
    keys = npy_data.keys()
    past_shape = npy_data['episodes'].shape

    folder_name = 'past_{}_step_{}_agent_{}_alpha_{}_number_{}/'.format(
        past_shape[1], past_shape[2], past_shape[0], alpha, number
    )
    for key in keys:
        if (key == 'dones') or (key == 'target_v'):
            continue
        np.save(folder_name + key, npy_data['key'])




