from environment.env import GridWorldEnv
from experiment2 import model
from experiment2.store_trajectories import Storage
from experiment2.config import get_configs

import torch.optim as optim
from utils import utils
from utils import dataset

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import ToMDataset
from utils.visualize import *
from utils.utils import *
from torch.utils.data import DataLoader



def train(tom_net, optimizer, train_loader, eval_loader, experiment_folder, dicts):

    for epoch in range(dicts['num_epoch']):
        results = tom_net.train(train_loader, optimizer)
        action_acc, comsumption_acc, action_loss, comsumption_loss, sr_loss, total_loss = results

        ev_results = evaluate(tom_net, eval_loader, experiment_folder, dicts)

        if epoch % dicts['save_freq'] == 0:
            save_path = utils.save_model(tom_net, dicts, experiment_folder, epoch)

        print('Train| Epoch {} Loss |Total {:.4f} Consume {:.4f} Action {:.4f} Value {:.4f}| Acc |Action {:.4f} Consume {:.4f}|'.format(epoch,
            total_loss, comsumption_loss, action_loss, sr_loss, action_acc, comsumption_acc))

        # TODO: ADD THE VISUALIZE PART
        # if epoch == 0:
        #     writer = SummaryWriter(os.path.join(save_path[:-12], 'logs'))
        #     visualizer = Visualizer(os.path.join(save_path[:-12], 'images'), grid_per_pixel=8,
        #                             max_epoch=num_epoch, height=env.height, width=env.width)
        #
        # for n in range(16):
        #     agent_xys = np.where(ev_past_trajectories[n, 0,:, :, :, 5] == 1)
        #     visualizer.get_past_traj(ev_past_trajectories[n][0][0], agent_xys, e, sample_num=n)
        #     visualizer.get_curr_state(ev_current_state[n], e, sample_num=n)
        #     visualizer.get_sr(ev_current_state[n], sr_preds[n], e, sample_num=n)
        #     visualizer.get_action(act_preds[n], e, sample_num=n)
        #     visualizer.get_prefer(con_preds[n], e, sample_num=n)

        # TODO: ADD THE TENSORBOARD
        # Print and Visualize the Results
        # writer.add_scalar('Train/Loss/Total_Loss', tot_losses[0], e)
        # writer.add_scalar('Train/Loss/Action_Loss', a_losses[0], e)
        # writer.add_scalar('Train/Loss/Consumed_Loss', c_losses[0], e)
        # writer.add_scalar('Train/Loss/SR_Loss', s_losses[0], e)
        # writer.add_scalar('Train/Acc/Action', a_accs[0], e)
        # writer.add_scalar('Train/Acc/Consumed', c_accs[0], e)
        #
        # writer.add_scalar('Eval/Loss/Total_Loss', ev_tot_loss, e)
        # writer.add_scalar('Eval/Loss/Action_Loss', ev_a_loss, e)
        # writer.add_scalar('Eval/Loss/Consumed_Loss', ev_c_loss, e)
        # writer.add_scalar('Eval/Loss/SR_Loss', ev_s_loss, e)
        # writer.add_scalar('Eval/Acc/Action', ev_action_acc, e)
        # writer.add_scalar('Eval/Acc/Consumed', ev_consume_acc, e)

        # msg = 'Train| Epoch {} Loss |Total {:.4f} Consume {:.4f} Action {:.4f} Value {:.4f}| Acc |Action {:.4f} Consume {:.4f}|'.format(\
        #     e, tot_losses[0], c_losses[0], a_losses[0], s_losses[0], a_accs[0], c_accs[0])
        # print(msg)

    return tom_net, save_path

def evaluate(tom_net, eval_loader, experiment_folder, dicts):

    with tr.no_grad():
         ev_results = tom_net.evaluate(eval_loader)
    # TODO : ADD THE VISUALIZE PART
    # TODO : ADD THE TENSORBOARD


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

    # Train
    optimizer = optim.Adam(tom_net.parameters(), lr=lr)
    train(tom_net, optimizer, train_loader, eval_loader, experiment_folder, dicts)

    # Test
    eval_storage.reset()
    test_data = eval_storage.extract()
    test_data['exp'] = 'exp2'
    test_dataset = dataset.ToMDataset(**test_data)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)
    evaluate(tom_net, test_loader, experiment_folder, dicts)