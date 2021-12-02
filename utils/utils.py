import torch as tr
import os
import datetime as dt
from dateutil.tz import gettz
import agent

def save_model(model, dicts):
    global year, month, day, hour, minutes, sec

    foldername = '{}_{}_{}_{}_{}_{}'.format(year, month, day, hour, minutes, sec)
    if not os.path.exists('./results/{}/checkpoints'.format(foldername)):
        os.makedirs('./results/{}/checkpoints'.format(foldername))
    dkeys = dicts.keys()
    filename_str = 'model_'
    for d in dkeys:
        vals = dicts[d]
        filename_str += '{}_{}_'.format(d, vals)

    tr.save(model, './results/{}/checkpoints/{}'.format(foldername, filename_str))

    return './results/{}/checkpoints'.format(foldername)

def make_folder():
    now = dt.datetime.now(gettz('Asia/Seoul'))
    year, month, day, hour, minutes, sec = str(now.year)[-2:], now.month, now.day, now.hour, now.minute, now.second

    foldername = '{}_{}_{}_{}_{}_{}'.format(year, month, day, hour, minutes, sec)
    folder_dir = './results/{}'.format(foldername)
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    return folder_dir

def make_pool(agent_type, move_penalty, alpha, num_agent=1000):
    agent_template = agent.agent_type[agent_type]
    population = []
    if (type(alpha) == int) or (type(alpha) == float):
        for _ in range(num_agent):
            population.append(agent_template(alpha=alpha, num_action=5))
    elif type(alpha) == list:
        for i, group in enumerate(num_agent):
            for _ in range(group):
                population.append(agent_template(alpha=alpha[i], num_action=5))
    else:
        assert ('Your alpha type is not proper type. We expect list, int or float. '
                'But we get the {}. Also check num_agent'.format(type(alpha)))
    return population
