from environment.env import GridWorldEnv
from experiment2 import model
from experiment2.store_trajectories import Storage
from experiment2.config import get_configs

import torch.optim as optim
from utils import utils
from utils import dataset
from utils import writer

from utils.visualize import *
from utils.utils import *
from torch.utils.data import DataLoader
import torch as tr
import numpy as np
import glob

def train(tom_net, optimizer, train_loader, eval_loaders, experiment_folder, writer, visualizer, dicts):
    for epoch in range(dicts['num_epoch']):
        results = tom_net.train(train_loader, optimizer)

        train_msg ='Train| Epoch {} Loss |Total {:.4f} Consume {:.4f} Action {:.4f} SR {:.4f}| Acc |Action {:.4f} Consume {:.4f}|'.format(
            epoch, results['total_loss'], results['consumption_loss'], results['action_loss'],
            results['sr_loss'], results['action_acc'], results['consumption_acc']
        )
        print(train_msg)
        writer.write(results, epoch, is_train=True)
        for e, eval_loader in enumerate(eval_loaders):
            ev_results = evaluate(tom_net, eval_loader)
            eval_msg = 'Eval{}| Epoch {} Loss |Total {:.4f} Consume {:.4f} Action {:.4f} SR {:.4f}| Acc |Action {:.4f} Consume {:.4f}|'.format(
                e, epoch, ev_results['total_loss'], ev_results['consumption_loss'], ev_results['action_loss'],
                ev_results['sr_loss'], ev_results['action_acc'], ev_results['consumption_acc'])

            print(eval_msg)
            writer.write(ev_results, epoch, is_train=False)

        if epoch % dicts['save_freq'] == 0:
            utils.save_model(tom_net, dicts, experiment_folder, epoch)



def evaluate(tom_net, eval_loader, visualizer=None, is_visualize=False,
             preference=None, mode='train'):
    '''
    we provide the base result of figure 2,
    but if you want to show the other results,
    run the inference.py after you have the models.
    '''
    with tr.no_grad():
        ev_results, ev_targs = tom_net.evaluate(eval_loader, is_visualize=is_visualize)
    if mode == 'train':
        filename=0
    else:
        filename=2
    if is_visualize:
        indiv_length = len(ev_results['past_traj'])
        for n in range(indiv_length):
            _, past_actions = np.where(ev_results['past_traj'][n, 0, :, :, :, 6:].sum((1, 2)) == 121)
            agent_xys = np.where(ev_results['past_traj'][n, 0, :, :, :, 5] == 1)
            visualizer.get_past_traj(ev_results['past_traj'][n][0][0], agent_xys, past_actions, filename, sample_num=n)
            visualizer.get_curr_state(ev_results['curr_state'][n], filename, sample_num=n)
            visualizer.get_action(ev_results['pred_actions'][n], filename, sample_num=n)
            visualizer.get_prefer(ev_results['pred_consumption'][n], filename, sample_num=n)
            visualizer.get_sr(ev_results['curr_state'][n], ev_results['pred_sr'][n], filename, sample_num=n)

            visualizer.get_action(ev_targs['targ_actions'][n], filename + 1, sample_num=n)
            visualizer.get_prefer(ev_targs['targ_consumption'][n], filename + 1, sample_num=n)
            visualizer.get_sr(ev_results['curr_state'][n], ev_targs['targ_sr'][n], filename + 1, sample_num=n)


        visualizer.get_consume_char(ev_results['e_char'], preference, filename)
        visualizer.tsne_consume_char(ev_results['e_char'], preference, filename)
    return ev_results


def run_experiment(num_epoch, main_experiment, sub_experiment, num_agent, batch_size, lr,
                   experiment_folder, alpha, save_freq, train_dir, eval_dir):


    exp_kwargs, env_kwargs, model_kwargs, agent_type = get_configs(sub_experiment)
    env = GridWorldEnv(env_kwargs)
    model_kwargs['num_agent'] = num_agent
    tom_net = model.PredNet(**model_kwargs)

    if model_kwargs['device'] == 'cuda':
        tom_net = tom_net.cuda()
    dicts = dict(main=main_experiment, sub=sub_experiment, alpha=alpha, batch_size=batch_size,
                 lr=lr, num_epoch=num_epoch, save_freq=save_freq)
    if train_dir != 'none':
        eval_dirs = glob.glob(eval_dir + '*')
        train_dataset = make_dataset(train_dir)
        eval_dataset_0 = make_dataset(train_dir)
        eval_dataset_1 = make_dataset(eval_dirs[0])
        eval_dataset_2 = make_dataset(eval_dirs[1])
        eval_dataset_3 = make_dataset(eval_dirs[2])
        eval_loader_0 = DataLoader(eval_dataset_0, batch_size=batch_size, shuffle=False)
        eval_loader_1 = DataLoader(eval_dataset_1, batch_size=batch_size, shuffle=False)
        eval_loader_2 = DataLoader(eval_dataset_2, batch_size=batch_size, shuffle=False)
        eval_loader_3 = DataLoader(eval_dataset_3, batch_size=batch_size, shuffle=False)
        eval_loaders = [eval_loader_0, eval_loader_1, eval_loader_2, eval_loader_3]

        train_prefer = np.load(train_dir + "/true_prefer.npy")
        test_1_prefer = np.load(eval_dirs[0] + "/true_prefer.npy")
        test_2_prefer = np.load(eval_dirs[1] + "/true_prefer.npy")
        test_3_prefer = np.load(eval_dirs[2] + "/true_prefer.npy")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    else:
        population = utils.make_pool(agent_type, exp_kwargs['move_penalty'], alpha, num_agent)
        # Make the Dataset
        train_storage = Storage(env, population, exp_kwargs['num_past'], exp_kwargs['num_step'])
        eval_storage = Storage(env, population[:1000], exp_kwargs['num_past'], exp_kwargs['num_step'])
        train_data = train_storage.extract()
        train_data['exp'] = 'exp2'
        eval_data = eval_storage.extract()
        eval_data['exp'] = 'exp2'
        train_dataset = dataset.ToMDataset(**train_data)
        eval_dataset = dataset.ToMDataset(**eval_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False) #len(eval_dataset)
        eval_loaders = [eval_loader]

    summary_writer = writer.Writer(os.path.join(experiment_folder, 'logs'))
    visualizer = Visualizer(os.path.join(experiment_folder, 'images'), grid_per_pixel=8,
                            max_epoch=num_epoch, height=env.height, width=env.width)
    # Train
    optimizer = optim.Adam(tom_net.parameters(), lr=lr)
    train(tom_net, optimizer, train_loader, eval_loaders, experiment_folder, summary_writer, visualizer, dicts)

    # Visualize Train
    train_fixed_loader = DataLoader(train_dataset, batch_size=1000, shuffle=False)
    train_prefer = train_storage.true_preference
    tr_results = evaluate(tom_net, train_fixed_loader, visualizer, is_visualize=True, preference=train_prefer)
    # Test
    eval_storage.reset()
    test_data = eval_storage.extract()
    test_data['exp'] = 'exp2'
    test_dataset = dataset.ToMDataset(**test_data)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    preference = eval_storage.true_preference
    ev_results = evaluate(tom_net, test_loader, visualizer, is_visualize=True, preference=preference, mode='eval')

def make_dataset(data_dir):
    data = {}
    data["episodes"] = np.load(data_dir + "/episodes.npy")
    data["curr_state"] = np.load(data_dir + "/curr_state.npy")
    data["target_action"] = np.load(data_dir + "/target_action.npy")
    data["target_prefer"] = np.load(data_dir + "/target_prefer.npy")
    data["target_sr"] = np.load(data_dir + "/target_sr.npy")
    data['exp'] = 'exp2'

    tom_dataset = dataset.ToMDataset(**data)

    return tom_dataset
