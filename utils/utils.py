import torch as tr
import os
import datetime as dt
from dateutil.tz import gettz

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
