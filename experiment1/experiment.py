import agent
import numpy as np
from environment.env import GridWorldEnv
import torch.optim as optim
from model.model1 import PredNet
from dateutil.tz import gettz
import datetime as dt
import torch as tr
from utils.dataset import ToMDatasetExp1
from experiment1.store_trajectories import Storage
from utils.visualize import *
from torch.utils.data import DataLoader

config = 'random'
agent = agent.agent_type[config]
now = dt.datetime.now(gettz('Asia/Seoul'))
year, month, day, hour, sec = str(now.year)[-2:], now.month, now.day, now.hour, now.second


def make_pool(num_agent, alpha):
    population = []
    if (type(alpha) == int) or (type(alpha) == float):
        for _ in range(num_agent):
            population.append(agent(alpha=alpha, num_action=5))
    elif type(alpha) == list:
        for i, group in enumerate(num_agent):
            for _ in range(group):
                population.append(agent(alpha=alpha[i], num_action=5))
    else:
        assert ('Your alpha type is not proper type. We expect list, int or float. '
                'But we get the {}. Also check num_agent'.format(type(alpha)))
    return population


def save_model(model, alpha, epoch):

    global year, month, day, hour, sec

    foldername = '{}_{}_{}_{}_{}'.format(year, month, day, hour, sec)
    if not os.path.exists('../results/checkpoints/{}'.format(foldername)):
        os.makedirs('../results/checkpoints/{}'.format(foldername))
    filename = 'model_alpha_{}_epoch_{}.pth'.format(alpha, epoch)
    tr.save(model, '../results/checkpoints/{}/{}'.format(foldername, filename))
    return '../results/checkpoints/{}'.format(foldername)


def tom_train(tom_nets, num_past, optims, env, alphas, num_epoch):
    for past in num_past:
        storage = Storage(env, population, past, step=1)
        past_trajectories, current_state, target_action, dones = storage.extract()
        losses = []
        accs = []
        for a in range(6):
            same_alpha_past_traj = past_trajectories[1000 * a:1000 * (a + 1)]
            same_alpha_curr_state = current_state[1000 * a:1000 * (a + 1)]
            same_alpha_target = target_action[1000 * a:1000 * (a + 1)]
            tom_dataset = ToMDatasetExp1(same_alpha_past_traj, same_alpha_curr_state, same_alpha_target)
            dataloader = DataLoader(tom_dataset, batch_size=16, shuffle=True, num_workers=8)
            tom_net = tom_nets[a]
            optim = optims[a]
            losses.append([])
            accs.append([])
            for e in range(num_epoch):
                acc, loss = tom_net.train(dataloader, optim)
                if e % 10 == 0:
                    save_path = save_model(tom_net, a, e)
                print(a, e, round(loss, 5), round(acc, 5))
                losses[a].append(loss)
                accs[a].append(acc)
        get_train_figure(losses, alphas, save_path, filename='loss_train_{}.jpg'.format(past))
        get_train_figure(accs, alphas, save_path, filename='accs_train_{}.jpg'.format(past))

    return tom_nets


def tom_evaluate(tom_nets, num_past, env, alphas):
    for past in num_past:
        storage = Storage(env, population, past, step=1)
        past_trajectories, current_state, target_action, dones = storage.extract()
        most_action_index, most_action_count = storage.get_most_act()
        losses = [[] for _ in range(6)]
        accs = [[] for _ in range(6)]
        for a in range(6):
            same_alpha_past_traj = past_trajectories[1000 * a:1000 * (a + 1)]
            same_alpha_curr_state = current_state[1000 * a:1000 * (a + 1)]
            same_alpha_target = target_action[1000 * a:1000 * (a + 1)]
            tom_dataset = ToMDatasetExp1(same_alpha_past_traj, same_alpha_curr_state, same_alpha_target)
            dataloader = DataLoader(tom_dataset, batch_size=1000, shuffle=False, num_workers=8)
            e_chars = []
            for i, tom_net in enumerate(tom_nets):
                acc, loss, e_char = tom_net.evaluate(dataloader)
                losses[i].append(loss)
                accs[i].append(acc)
                e_chars.append(e_char.detach().cpu().numpy())
        e_chars = np.concatenate(e_chars, axis=0)
        foldername = '{}_{}_{}_{}_{}'.format(year, month, day, hour, sec)
        save_path = '../results/checkpoints/{}'.format(foldername)
        get_test_figure(losses, alphas, save_path, filename='loss_eval_{}.jpg'.format(past))
        get_test_figure(accs, alphas, save_path, filename='accs_eval_{}.jpg'.format(past))
        visualize_embedding(e_chars, most_action_index, most_action_count, save_path,
                            filename='e_char_{}.jpg'.format(past))


if __name__ == '__main__':
    tom_nets = [PredNet(num_past=1, num_input=11, device='cuda').cuda()
                for _ in range(6)]
    optims = [optim.Adam(tom_nets[i].parameters(), lr=1e-4)
              for i in range(6)]

    # Since there are no prefer, we do not select preference
    env_config = dict(height=11, width=11, pixel_per_grid=8,
                      preference=100, exp=1, save=True)
    population = make_pool([1000] * 6, [0.01, 0.03, 0.1, 0.3, 1, 3])
    num_past = np.arange(10, 11)
    env = GridWorldEnv(env_config)
    num_epoch = 100
    alphas = [0.01, 0.03, 0.1, 0.3, 1, 3]

    tom_nets = tom_train(tom_nets, num_past, optims, env, alphas, num_epoch)
    tom_evaluate(tom_nets, num_past, env, alphas)

