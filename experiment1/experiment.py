from environment.env import GridWorldEnv
from experiment1 import model
from experiment1.store_trajectories import Storage
from experiment1.config import get_configs

from utils.visualize import *
from utils import utils
from utils import dataset
from utils import writer

from torch.utils.data import DataLoader
import torch as tr
import torch.optim as optim
import numpy as np

def train(tom_net, optimizer, train_loader, eval_loader, experiment_folder, writer, dicts):

    for epoch in range(dicts['num_epoch']):
        results = tom_net.train(train_loader, optimizer)

        ev_results = evaluate(tom_net, eval_loader)

        if epoch % dicts['save_freq'] == 0:
            utils.save_model(tom_net, dicts, experiment_folder, epoch)
        writer.write(results, epoch, is_train=True)
        writer.write(ev_results, epoch, is_train=False)
        print('Train| Epoch {} Loss |{:.4f}|Acc |{:.4f}'.format(epoch, results['action_loss'], results['action_acc']))
        print('Eval| Epoch {} Loss |{:.4f}|Acc |{:.4f}'.format(epoch, ev_results['action_loss'], ev_results['action_acc']))

        # TODO: ADD THE VISUALIZE PART



def evaluate(tom_net, eval_loader, visualizer=None, is_visualize=False,
             most_act=None, count_act=None):
    '''
    we provide the base result of figure 2,
    but if you want to show the other results,
    run the inference.py after you have the models.
    '''
    with tr.no_grad():
        ev_results = tom_net.evaluate(eval_loader, is_visualize=is_visualize)

    if is_visualize:
        for n in range(16):
            agent_xys = np.where(ev_results['past_traj'][n, 0, :, :, :, 5] == 1)
            visualizer.get_past_traj(ev_results['past_traj'][n][0][0], agent_xys, 0, sample_num=n)
            visualizer.get_curr_state(ev_results['curr_state'][n], 0, sample_num=n)
            visualizer.get_action(ev_results['pred_actions'][n], 0, sample_num=n)

        visualizer.get_char(ev_results['e_char'], most_act, count_act, 0)
    return ev_results

def run_experiment(num_epoch, main_experiment, sub_experiment, num_agent, batch_size, lr,
                   experiment_folder, alpha, save_freq):

    exp_kwargs, env_kwargs, model_kwargs, agent_kwargs = get_configs(sub_experiment)
    population = utils.make_pool('random', exp_kwargs['move_penalty'], alpha, num_agent)
    env = GridWorldEnv(env_kwargs)
    tom_net = model.PredNet(**model_kwargs)

    if model_kwargs['device'] == 'cuda':
        tom_net = tom_net.cuda()

    dicts = dict(main=main_experiment, sub=sub_experiment, alpha=alpha, batch_size=batch_size,
                 lr=lr, num_epoch=num_epoch, save_freq=save_freq)

    # Make the Dataset
    train_storage = Storage(env, population, exp_kwargs['num_past'], exp_kwargs['num_step'])
    eval_storage = Storage(env, population, exp_kwargs['num_past'], exp_kwargs['num_step'])
    train_data = train_storage.extract()
    train_data['exp'] = 'exp1'
    eval_data = eval_storage.extract()
    eval_data['exp'] = 'exp1'
    train_dataset = dataset.ToMDataset(**train_data)
    eval_dataset = dataset.ToMDataset(**eval_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)

    summary_writer = writer.Writer(os.path.join(experiment_folder, 'logs'))
    visualizer = Visualizer(os.path.join(experiment_folder, 'images'), grid_per_pixel=8,
                            max_epoch=num_epoch, height=env.height, width=env.width)

    # Train
    optimizer = optim.Adam(tom_net.parameters(), lr=lr)
    train(tom_net, optimizer, train_loader, eval_loader, experiment_folder,
          summary_writer, dicts)

    # Test
    eval_storage.reset()
    test_data = eval_storage.extract()
    test_data['exp'] = 'exp1'
    test_dataset = dataset.ToMDataset(**test_data)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    most_act, count_act = eval_storage.get_most_act()
    ev_results = evaluate(tom_net, test_loader, visualizer, is_visualize=True,
                          most_act=most_act, count_act=count_act)



