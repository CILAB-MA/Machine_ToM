import numpy as np
import torch as tr
import os
import datetime as dt
from dateutil.tz import gettz
import agent

def save_model(model, dicts, experiment_folder, epoch):
    model_path = '{}/checkpoints'.format(experiment_folder)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    dkeys = dicts.keys()
    filename_str = 'model_'
    for d in dkeys:
        vals = dicts[d]
        filename_str += '{}_{}_'.format(d, vals)
    filename_str += 'epoch_{}'.format(epoch)
    tr.save(model, '{}/{}'.format(model_path, filename_str))

def make_folder():
    now = dt.datetime.now(gettz('Asia/Seoul'))
    year, month, day, hour, minutes, sec = str(now.year)[-2:], now.month, now.day, now.hour, now.minute, now.second

    foldername = '{}_{}_{}_{}_{}_{}'.format(year, month, day, hour, minutes, sec)
    folder_dir = './results/{}'.format(foldername)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    return folder_dir

def make_pool(agent_type, move_penalty, alpha, num_agent):
    agent_template = agent.agent_type[agent_type]
    population = []

    if (type(alpha) == int) or (type(alpha) == float):
        for _ in range(num_agent):
            population.append(agent_template(alpha=alpha, num_action=5, move_penalty=move_penalty))
    elif type(alpha) == np.ndarray:
        num_agent_list = [num_agent] * len(alpha)
        for idx_alpha, num in enumerate(num_agent_list):
            for _ in range(num):
                population.append(agent_template(alpha=alpha[idx_alpha], num_action=5, move_penalty=move_penalty))
    else:
        assert ('Your alpha type is not proper type. We expect list, int or float. '
                'But we get the {}. Also check num_agent'.format(type(alpha)))
    np.random.shuffle(population)
    return population
