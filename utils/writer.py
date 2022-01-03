import os
from torch.utils.tensorboard import SummaryWriter

class Writer:
    def __init__(self, experiment_folder):
        self.writer = SummaryWriter(os.path.join(experiment_folder, 'logs'))

    def write(self, dicts, epoch, is_train=True, num_eval=None):
        dkeys = dicts.keys()

        if is_train:
            model_mode = 'Train'
        else:
            model_mode = 'Eval' + str(num_eval)

        for key in dkeys:
            val = dicts[key]
            self.writer.add_scalar('{}/{}'.format(key, model_mode), val, epoch)