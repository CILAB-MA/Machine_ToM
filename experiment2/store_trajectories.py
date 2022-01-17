import numpy as np
import copy
from tqdm import tqdm

class Storage(object):

    def __init__(self, env, population, num_past, num_step):
        self.env = env
        self.past_trajectories = np.zeros([len(population), num_past, num_step, env.height, env.width, 11], dtype=np.float16)
        self.current_state = np.zeros([len(population), env.height, env.width, 6], dtype=np.float16)
        self.target_action = np.zeros([len(population), 1], dtype=np.int16)
        self.target_preference = np.full([len(population), 1], 4, dtype=np.float16)
        self.true_preference = np.zeros([len(population), 4], dtype=np.float16)
        self.target_sr = np.zeros([len(population), env.height, env.width, 3], dtype=np.float16)
        self.dones = np.zeros([len(population), num_past, num_step, 1], dtype=np.int16)
        self.population = population
        self.num_past = num_past
        self.action_count = np.zeros([len(population), 5], dtype=np.int16)
        self.num_step = num_step

    def extract(self, custom_past=-100, custom_query=-100, slicing=None):
        if slicing != None:
            population = self.population[:slicing]
        else:
            population = self.population
        print('Storage', slicing, len(population))
        for agent_index, agent in tqdm(enumerate(population), total=len(population)):
            self.env.prefer_reward = agent.reward
            self.env.preference = agent.most_prefer
            self.true_preference[agent_index] = agent.reward
            for past_epi in range(self.num_past):
                self.env.num_wall = np.random.randint(5)
                if np.sum(custom_past) > 0:
                    obs = self.env.reset(custom=custom_past[agent_index][past_epi], wall=True)
                else:
                    obs = self.env.reset(wall=True)
                #self.env.obs_well_show()
                agent.train(obs)
                # gathering past trajectories
                for step in range(self.num_step):
                    action = agent.act(obs)
                    self.action_count[agent_index, action] += 1
                    spatial_concat_action = np.zeros((self.env.height, self.env.width, 5))
                    spatial_concat_action[:, :,  action] = 1
                    obs_concat = np.concatenate([obs, spatial_concat_action], axis=-1)
                    self.past_trajectories[agent_index, past_epi, step] = obs_concat
                    self.dones[agent_index, past_epi, step] = 1

                    obs, reward, done, info = self.env.step(action)
                    if done:
                        # 0 = done
                        break
            # gathering current_state
            consumed = None
            done = False
            self.env.num_wall = np.random.randint(5)
            while done != True:
                if np.sum(custom_query) > 0:
                    curr_obs = self.env.reset(custom=custom_query[agent_index], wall=True)
                else:
                    curr_obs = self.env.reset(wall=True)
                #self.env.obs_well_show()
                agent.train(curr_obs)
                target_action = agent.act(curr_obs)
                action = copy.deepcopy(target_action)
                sr = np.zeros((self.env.height, self.env.width, 3))
                sr[self.env.agent_xy[0], self.env.agent_xy[1], :] = 1
                gamma = np.array([0.5, 0.9, 0.99])
                for s in range(self.env.epi_max_step):
                    obs, _, done, info = self.env.step(action)
                    sr[self.env.agent_xy[0], self.env.agent_xy[1], :] = gamma ** (s + 1)
                    action = agent.act(obs)
                    if done:
                        consumed = info['consumed']
                        break
            # self.env.obs_well_show()
            #print(curr_obs.sum(-1))
            sr[:, :, 0] /= np.sum(sr[:, :, 0])
            sr[:, :, 1] /= np.sum(sr[:, :, 1])
            sr[:, :, 2] /= np.sum(sr[:, :, 2])
            self.current_state[agent_index] = curr_obs
            self.target_action[agent_index] = target_action
            if consumed != None:
                self.target_preference[agent_index] = consumed
            self.target_sr[agent_index] = sr
        #print('Past Traj : {} Curr state : {} Targ action : {} Targ goal : {} Targ SR : {} Targ Done : {}'.format(
        #    self.past_trajectories.shape, self.current_state.shape, self.target_action.shape, self.target_preference.shape,
        #    self.target_sr.shape, self.target_policy.shape, self.dones.shape
        #))
        return dict(episodes=self.past_trajectories[:len(population)], curr_state=self.current_state[:len(population)],
                    target_action=self.target_action[:len(population)], target_prefer=self.target_preference[:len(population)],
                    target_sr=self.target_sr[:len(population)], true_prefer=self.true_preference[:len(population)])

    def reset(self):
        self.past_trajectories = np.zeros(self.past_trajectories.shape)
        self.current_state = np.zeros(self.current_state.shape)
        self.target_action = np.zeros(self.target_action.shape)

        self.target_preference = np.zeros(self.target_preference.shape)
        self.target_sr = np.zeros(self.target_sr.shape)

        self.action_count = np.zeros(self.action_count.shape)


    def get_most_act(self):
        action_count = copy.deepcopy(self.action_count)
        action_count /= np.reshape(np.sum(action_count, axis=-1), (-1, 1))

        return np.argmax(action_count, axis=-1), np.max(action_count, axis=-1)







if __name__ == '__main__':
    from environment.env import GridWorldEnv
    import agent
    env_config = dict(height=25, width=25, pixel_per_grid=8, preference=100, exp=2, save=True)
    env = GridWorldEnv(env_config)
    agent_config = dict(name='reward_seeking', species=[0.01], num=1000)
    agent = agent.agent_type[agent_config['name']]
    print(env.num_wall)
    storage = Storage(env, population=[agent(alpha=0.01, num_action=5, move_penalty=-0.01)], num_past=1, num_step=1)
