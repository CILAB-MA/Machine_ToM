import agent as agent
from environment.env import GridWorldEnv
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import ToMDataset
from utils.visualize import *
from utils.utils import *

from experiment2.store_trajectories import Storage
from experiment2 import config, model

from torch.utils.data import DataLoader


def make_pool(num_agent, alpha, move_penalty):
    population = []
    if (type(alpha) == int) or (type(alpha) == float):
        for _ in range(num_agent):
            population.append(agent(alpha=alpha, num_action=5, move_penalty=move_penalty))
    elif type(alpha) == list:
        for i, group in enumerate(num_agent):
            for _ in range(group):
                population.append(agent(alpha=alpha[i], num_action=5, move_penalty=move_penalty))
    else:
        assert ('Your alpha type is not proper type. We expect list, int or float. '
                'But we get the {}. Also check num_agent'.format(type(alpha)))
    return population


def train(tom_nets, optims, env, population, is_active, method_type,
              num_past, num_step, num_epoch, batch_size, dicts):

    storage = Storage(env, population, num_past, num_step=num_step)
    ev_storage = Storage(env, population, num_past, num_step=num_step)
    for e in range(num_epoch):
        past_trajectories, current_state, target_action, target_prefer, target_sr, target_v, dones = storage.extract()
        tom_dataset = ToMDataset(past_trajectories, current_state, target_action, target_prefer,
                                 target_sr, target_v, dones, 'exp2')

        ev_past_trajectories, ev_current_state, ev_target_action, ev_target_prefer, ev_target_sr, ev_target_v, ev_dones = ev_storage.extract()
        ev_tom_dataset = ToMDataset(ev_past_trajectories, ev_current_state, ev_target_action,
                                 ev_target_prefer, ev_target_sr, ev_target_v, dones, 'exp2')
        ev_dataloader = DataLoader(ev_tom_dataset, batch_size=len(ev_tom_dataset), shuffle=False)

        dataloader = DataLoader(tom_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

        a_accs = []
        c_accs = []

        tot_losses = []
        c_losses = []
        a_losses = []
        s_losses = []
        for tom_net, optim in zip(tom_nets, optims):
            action_acc, consume_acc, a_loss, c_loss, s_loss, tot_loss = tom_net.train(dataloader, optim)

            c_losses.append(c_loss)
            a_losses.append(a_loss)
            s_losses.append(s_loss)
            tot_losses.append(tot_loss)
            a_accs.append(action_acc)
            c_accs.append(consume_acc)

        best_ind = np.argmax(tot_losses)
        with tr.no_grad():
            ev_results = tom_nets[best_ind].evaluate(ev_dataloader)
        ev_action_acc, ev_consume_acc, ev_a_loss, ev_c_loss, ev_s_loss, ev_tot_loss, sr_preds, con_preds, act_preds = ev_results

        if e % 10 == 0:
            save_path = save_model(tom_nets[best_ind], dicts)
            if e == 0:
                writer = SummaryWriter(os.path.join(save_path[:-12], 'logs'))
                visualizer = Visualizer(os.path.join(save_path[:-12], 'images'), grid_per_pixel=8,
                                        max_epoch=num_epoch, height=env.height, width=env.width)

            for n in range(16):
                agent_xys = np.where(ev_past_trajectories[n, 0,:, :, :, 5] == 1)
                visualizer.get_past_traj(ev_past_trajectories[n][0][0], agent_xys, e, sample_num=n)
                visualizer.get_curr_state(ev_current_state[n], e, sample_num=n)
                visualizer.get_sr(ev_current_state[n], sr_preds[n], e, sample_num=n)
                visualizer.get_action(act_preds[n], e, sample_num=n)
                visualizer.get_prefer(con_preds[n], e, sample_num=n)

        ## Print and Visualize the Results
        writer.add_scalar('Train/Loss/Total_Loss', tot_losses[0], e)
        writer.add_scalar('Train/Loss/Action_Loss', a_losses[0], e)
        writer.add_scalar('Train/Loss/Consumed_Loss', c_losses[0], e)
        writer.add_scalar('Train/Loss/SR_Loss', s_losses[0], e)
        writer.add_scalar('Train/Acc/Action', a_accs[0], e)
        writer.add_scalar('Train/Acc/Consumed', c_accs[0], e)

        writer.add_scalar('Eval/Loss/Total_Loss', ev_tot_loss, e)
        writer.add_scalar('Eval/Loss/Action_Loss', ev_a_loss, e)
        writer.add_scalar('Eval/Loss/Consumed_Loss', ev_c_loss, e)
        writer.add_scalar('Eval/Loss/SR_Loss', ev_s_loss, e)
        writer.add_scalar('Eval/Acc/Action', ev_action_acc, e)
        writer.add_scalar('Eval/Acc/Consumed', ev_consume_acc, e)

        msg = 'Train| Epoch {} Loss |Total {:.4f} Consume {:.4f} Action {:.4f} Value {:.4f}| Acc |Action {:.4f} Consume {:.4f}|'.format(\
            e, tot_losses[0], c_losses[0], a_losses[0], s_losses[0], a_accs[0], c_accs[0])
        print(msg)

        # Visualize the Training


    return tom_nets, save_path

def evaluate(tom_net, env, population, save_path,
                 num_past, num_step, num_iteration, batch_size, dicts):

    storage = Storage(env, population, num_past, num_step=num_step)
    past_trajectories, current_state, target_action, dones = storage.extract()
    most_action_index, most_action_count = storage.get_most_act()
    losses = 0
    accs = 0
    for e in range(num_iteration):
        past_trajectories, current_state, target_action, target_prefer, target_sr, dones = storage.extract()
        tom_dataset = ToMDataset(past_trajectories, current_state, target_action, target_prefer,
                                 target_sr, dones, 'exp2')
        dataloader = DataLoader(tom_dataset, batch_size=batch_size, shuffle=True, num_workers=8)


        a, l = tom_net.evaluate(dataloader)
        losses += l
        accs += a


    loss_mu = losses / num_iteration
    accs_mu = accs / num_iteration

    get_test_figure(loss_mu, dicts, save_path, filename='loss')
    get_test_figure(accs_mu, dicts, save_path, filename='acc')

    # TODO : visualize other measures

    # visulaize_embedding(e_chars, most_action_index, most_action_count, save_path,
    # filename='e_char_{}.jpg'.format(past))

# def make_pool(sub_experiment):
#
#     agent_config = dict(name='reward_seeking', species=[0.01], num=1000)
#     agent = agent.agent_type[agent_config['name']]



def run_experiment(num_epoch, sub_experiment, batch_size, lr, num_eval, experiment_folder):

    exp_kwargs, env_kwargs, model_kwargs = config.get_configs(sub_experiment)

    population = make_pool(sub_experiment, exp_kwargs['move_penalty'])

    env = GridWorldEnv(env_kwargs)

    tom_net = model.PredNet(**model_kwargs)
    if model_kwargs['device'] == 'cuda':
        tom_net = tom_net.cuda()

    storage = Storage()
    optimizer = optim.Adam(tom_net)
    train(optimizer, tom_net, storage, num_epoch, batch_size, lr, num_eval, experiment_folder)
    evaluate(tom_net, storage, num_epoch, batch_size, lr, num_eval, experiment_folder)