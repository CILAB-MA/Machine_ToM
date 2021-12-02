from environment.env import GridWorldEnv
from experiment1 import model
from experiment1.store_trajectories import Storage
from experiment1.config import get_configs

from utils.visualize import *
from utils import utils
from utils import dataset

from torch.utils.data import DataLoader
import torch as tr
import torch.optim as optim

def train(tom_net, optimizer, train_loader, eval_loader, dicts):

    for epoch in range(dicts['num_epoch']):
        results = tom_net.train(train_loader, optimizer)
        acc, loss = results

        ev_results = evaluate(tom_net, eval_loader, dicts)

        if epoch % dicts['save_freq'] == 0:
            save_path = utils.save_model(tom_net, )

    # TODO: ADD THE VISUALIZE PART
    # TODO: ADD THE TENSORBOARD

def evaluate(tom_net, eval_loader, dicts):

    with tr.no_grad():
         ev_res = tom_net.evaluate(eval_loader)

    # TODO : ADD THE VISUALIZE PART
    # TODO : ADD THE TENSORBOARD

def run_experiment(num_epoch, main_experiment, sub_experiment, batch_size, lr,
                   num_eval, experiment_folder, alpha):

    exp_kwargs, env_kwargs, model_kwargs, agent_kwargs = get_configs(sub_experiment)
    population = utils.make_pool(sub_experiment, exp_kwargs['move_penalty'], alpha)
    env = GridWorldEnv(env_kwargs)
    tom_net = model.PredNet(**model_kwargs)

    if model_kwargs['device'] == 'cuda':
        tom_net = tom_net.cuda()
    dicts = dict(main=main_experiment, sub=sub_experiment, alpha=alpha, batch_size=batch_size,
                 lr=lr, num_epoch=num_epoch)

    # Make the Dataset
    train_storage = Storage(env)
    eval_storage = Storage(env)
    train_data = train_storage.extract()
    train_data['exp'] = 'exp1'
    eval_data = eval_storage.extract()
    eval_data['exp'] = 'exp1'
    train_dataset = dataset.ToMDataset(**train_data)
    eval_dataset = dataset.ToMDataset(**eval_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=len(eval_dataset), shuffle=False)

    # Train
    optimizer = optim.Adam(tom_net.parameters(), lr=lr)
    train(tom_net, optimizer, population, train_loader, eval_loader, dicts)

    # Test
    eval_storage.reset()
    test_data = eval_storage.extract()
    test_data['exp'] = 'exp1'
    test_dataset = dataset.ToMDataset(**test_data)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    evaluate(tom_net, test_loader, dicts)



