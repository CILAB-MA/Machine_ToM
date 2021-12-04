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


def train(tom_net, optimizer, train_loader, eval_loader, experiment_folder, writer, visualizer, dicts):

    for epoch in range(dicts['num_epoch']):
        results = tom_net.train(train_loader, optimizer)
        action_acc, comsumption_acc, action_loss, comsumption_loss, sr_loss, total_loss = results

        ev_results = evaluate(tom_net, eval_loader, experiment_folder, dicts)

        if epoch % dicts['save_freq'] == 0:
            utils.save_model(tom_net, dicts, experiment_folder, epoch)
        writer.write(results, epoch, is_train=True)
        writer.write(ev_results, epoch, is_train=False)

        train_msg ='Train| Epoch {} Loss |Total {:.4f} Consume {:.4f} Action {:.4f} SR {:.4f}| Acc |Action {:.4f} Consume {:.4f}|'.format(
            epoch, results['total_loss'], results['consumption_loss'], results['action_loss'],
            results['sr_loss'], results['action_acc'], results['consumption_acc']
        )
        eval_msg ='Eval| Epoch {} Loss |Total {:.4f} Consume {:.4f} Action {:.4f} SR {:.4f}| Acc |Action {:.4f} Consume {:.4f}|'.format(
            epoch, ev_results['total_loss'], ev_results['consumption_loss'], ev_results['action_loss'],
            ev_results['sr_loss'], ev_results['action_acc'], ev_results['consumption_acc']
        )
        print(train_msg)
        print(eval_msg)
        # TODO: ADD THE VISUALIZE PART


def evaluate(tom_net, eval_loader, experiment_folder, dicts):

    with tr.no_grad():
         ev_results = tom_net.evaluate(eval_loader)

    return ev_results


def run_experiment(num_epoch, main_experiment, sub_experiment, batch_size, lr,
                   experiment_folder, alpha, save_freq):

    exp_kwargs, env_kwargs, model_kwargs, agent_type = get_configs(sub_experiment)

    population = utils.make_pool(agent_type, exp_kwargs['move_penalty'], alpha)
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
    train_data['exp'] = 'exp2'
    eval_data = eval_storage.extract()
    eval_data['exp'] = 'exp2'
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
          summary_writer, visualizer, dicts)

    # Test
    eval_storage.reset()
    test_data = eval_storage.extract()
    test_data['exp'] = 'exp2'
    test_dataset = dataset.ToMDataset(**test_data)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    ev_results = evaluate(tom_net, test_loader, experiment_folder, dicts)
    print(ev_results)