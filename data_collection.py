import argparse
import sys


from utils import utils
from utils import dataset
from environment.env import GridWorldEnv

from experiment2.store_trajectories import Storage
from experiment2.config import get_configs
import numpy as np
import os
import copy

def get_bool(args):
    return eval(args)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--number', '-n', type=int, default=17)
    parser.add_argument('--num_agent', '-na', type=int, default=5)
    parser.add_argument('--main_exp', '-me', type=int, default=2)
    parser.add_argument('--sub_exp', '-se', type=int, default=1)
    parser.add_argument('--alpha', '-a', type=float, default=0.01)
    parser.add_argument('--base_dir', '-b', type=str, default='./data')
    parser.add_argument('--slicing', '-s', type=int, default=200)
    parser.add_argument('--is_wall', '-w', type=get_bool, default=True)
    args = parser.parse_args()
    return args


class DataCollector(object):

    def __init__(self, args):
        '''

        Train : num_past 1 or 3

        Eval :
            1) Diff Population / New Past Traj / New Curr State
            2) Same Population / New Past Traj / New Curr State
            3) Same Population / Same Past Traj / New urr State
        '''

        exp_kwargs, env_kwargs, model_kwargs, agent_type = get_configs(num_exp=1)
        print('is_wall', args.is_wall)
        # make settings
        if args.main_exp == 2:
            self.population = utils.make_pool(agent_type, exp_kwargs['move_penalty'], args.alpha, args.num_agent)
            self.diff_population = utils.make_pool(agent_type, exp_kwargs['move_penalty'], args.alpha, int(args.num_agent / 5))
            foldername = 'exp_{}_sub_{}_agent_{}_id_{}'.format(args.main_exp, args.sub_exp,
                                                                         args.num_agent, args.number)

        self.env = GridWorldEnv(env_kwargs)

        if args.sub_exp == 2:
            self.fixed_mdp = self.env.reset(wall=args.is_wall).reshape((1, env_kwargs['height'], env_kwargs['width'], 6))
            empty_list = get_empty(self.fixed_mdp)

            self.train_past = np.zeros((len(self.population), 1, env_kwargs['height'], env_kwargs['width'], 6))
            self.eval1_past = np.zeros((int(len(self.population)/5), 1, env_kwargs['height'], env_kwargs['width'], 6))
            self.eval2_past = np.zeros((int(len(self.population)/5), 1, env_kwargs['height'], env_kwargs['width'], 6))

            for num_past_i in range(exp_kwargs['num_past']):
                # make train past mdp
                train_past = get_new_loc(self.fixed_mdp, empty_list)
                for i in range(len(self.population)-1):
                    train_past = np.append(train_past, get_new_loc(self.fixed_mdp, empty_list), axis=0)
                self.train_past = np.append(self.train_past, train_past.reshape((len(self.population), 1, env_kwargs['height'], env_kwargs['width'], 6)), axis=1)

                # make eval past mdp
                eval1_past = get_new_loc(self.fixed_mdp, empty_list)
                eval2_past = get_new_loc(self.fixed_mdp, empty_list)
                for i in range(args.slicing - 1):
                    eval1_past = np.append(eval1_past, get_new_loc(self.fixed_mdp, empty_list), axis=0)
                    eval2_past = np.append(eval2_past, get_new_loc(self.fixed_mdp, empty_list), axis=0)
                print(self.eval1_past.shape)

                self.eval1_past = np.append(self.eval1_past, eval1_past.reshape((int(len(self.population)/5), 1, env_kwargs['height'], env_kwargs['width'], 6)), axis=1)
                self.eval2_past = np.append(self.eval2_past, eval2_past.reshape((int(len(self.population)/5), 1, env_kwargs['height'], env_kwargs['width'], 6)), axis=1)
            self.train_past = self.train_past[:, 1:, :, :, :]
            self.eval1_past = self.eval1_past[:, 1:, :, :, :]
            self.eval2_past = self.eval2_past[:, 1:, :, :, :]

            # make train eval mdp
            self.train_query = get_new_loc(self.fixed_mdp, empty_list)
            for i in range(len(self.population) - 1):
                self.train_query = np.append(self.train_query, get_new_loc(self.fixed_mdp, empty_list), axis=0)

            # make eval query mdp
            self.eval1_query = get_new_loc(self.fixed_mdp, empty_list)
            self.eval2_query = get_new_loc(self.fixed_mdp, empty_list)
            self.eval3_query = get_new_loc(self.fixed_mdp, empty_list)
            for i in range(args.slicing - 1):
                self.eval1_query = np.append(self.eval1_query, get_new_loc(self.fixed_mdp, empty_list), axis=0)
                self.eval2_query = np.append(self.eval2_query, get_new_loc(self.fixed_mdp, empty_list), axis=0)
                self.eval3_query = np.append(self.eval3_query, get_new_loc(self.fixed_mdp, empty_list), axis=0)

        self.storage = Storage(self.env, self.population, exp_kwargs['num_past'], exp_kwargs['num_step'])
        self.diff_storage = Storage(self.env, self.diff_population, exp_kwargs['num_past'], exp_kwargs['num_step'])
        self.slicing = args.slicing

        self.base_dir = os.path.join(args.base_dir, foldername)

    def make_train_set(self):
        if args.sub_exp == 2:
            self.tr_data = self.storage.extract(custom_past=self.train_past, custom_query=self.train_query)
        else:
            self.tr_data = self.storage.extract()
        dataset.save_data(self.tr_data, 'train', self.base_dir)

    def make_eval_set(self):
        # make first eval set
        if args.sub_exp == 2:
            eval_type1_data = self.diff_storage.extract(custom_past=self.eval1_past, custom_query=self.eval1_query)
        else:
            eval_type1_data = self.diff_storage.extract()
        dataset.save_data(eval_type1_data, 'eval', self.base_dir, eval=1)
        print(len(eval_type1_data['episodes']))

        # make third eval set
        if args.sub_exp == 2:
            eval_type2_data = self.storage.extract(custom_past=self.train_past, custom_query=self.eval3_query,
                                                   slicing=self.slicing)
        else:
            eval_type2_data = self.storage.extract(custom_past=self.tr_data['episodes'][:, :, 0, :, :, :6],
                                               slicing=self.slicing)
        eval_type2_data['true_prefer'] = self.tr_data['true_prefer']
        dataset.save_data(eval_type2_data, 'eval', self.base_dir, eval=3)
        print(len(eval_type2_data['episodes']))

        # make second eval set
        self.storage.reset()
        if args.sub_exp == 2:
            eval_type3_data = self.storage.extract(custom_past=self.eval2_past, custom_query=self.eval2_query,
                                                   slicing=self.slicing)
        else:
            eval_type3_data = self.storage.extract(slicing=self.slicing)
        eval_type3_data['true_prefer'] = self.tr_data['true_prefer']
        dataset.save_data(eval_type3_data, 'eval', self.base_dir, eval=2)
        print(len(eval_type3_data['episodes']))


def get_empty(obs):
    obs = copy.deepcopy(obs[0])
    obs = obs.sum(axis=-1)
    xs, ys = np.where(obs == 0)
    empty_list = [[x, y] for x, y in zip(xs, ys)]
    return empty_list


def get_new_loc(obs, empty_list):
    obs = copy.deepcopy(obs)
    idx = np.random.choice(len(empty_list), 5, replace=False)
    xys = np.array(empty_list)[idx]

    for i, xy in enumerate(xys):
        obs[0, :, :, i + 1] = 0
        obs[0, xy[0], xy[1], i + 1] = 1
    return obs


if __name__ == '__main__':
    args = parse_args()
    collector = DataCollector(args)
    collector.make_train_set()
    collector.make_eval_set()