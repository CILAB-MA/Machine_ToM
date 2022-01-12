from torch.utils.data import Dataset
import numpy as np
import os

class ToMDataset(Dataset):

    def __init__(self, episodes, curr_state, target_action, target_prefer=None,
                 target_sr=None, target_value=None, true_prefer=None, exp='exp1'):
        self.episodes = episodes
        self.curr_state = curr_state

        self.target_action = target_action
        self.target_prefer = target_prefer
        self.target_sr = target_sr

        self.exp = exp

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, ind):
        if self.exp == 'exp1':
            return self.episodes[ind], self.curr_state[ind], self.target_action[ind]
        else:
            return self.episodes[ind], self.curr_state[ind], self.target_action[ind],\
                   self.target_prefer[ind], self.target_sr[ind], ind


def save_data(npy_data, is_train, base_dir, eval=None):
    keys = npy_data.keys()
    past_shape = npy_data['episodes'].shape

    folder_name = os.path.join(base_dir, is_train)
    if eval != None:
        folder_name = os.path.join(folder_name, str(eval))
    make_dirs(folder_name)

    for key in keys:
        if (key == 'dones') or (key == 'target_v'):
            continue
        np.save(folder_name + '/' + key, npy_data[key])


def make_dirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise