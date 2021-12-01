import numpy as np
import cv2
import copy


class Storage(object):

    def __init__(self, env, population, num_past, num_step):
        self.env = env
        self.past_trajectories = np.zeros([len(population), num_past, num_step, env.height, env.width, 11])
        self.current_state = np.zeros([len(population), env.height, env.width, 6])
        self.target_action = np.zeros([len(population), 1])
        self.target_preference = np.zeros([len(population), 4])
        self.target_sr = np.zeros([len(population), env.height, env.width, 3])
        self.target_policy = np.zeros([len(population), env.height, env.width, 1])
        self.dones = np.zeros([len(population), num_past, num_step, 1])
        self.population = population
        self.num_past = num_past
        self.action_count = np.zeros([len(population), 5])
        self.num_step = num_step

    def extract(self, custom_past=-100, custom_query=-100):
        for agent_index, agent in enumerate(self.population):
            self.env.prefer_reward = agent.reward
            self.env.preference = agent.most_prefer
            for past_epi in range(self.num_past):
                self.env.num_wall = np.random.randint(5)
                if np.sum(custom_past) > 0:
                    obs = self.env.reset(custom=custom_past[agent_index], wall=True)
                else:
                    obs = self.env.reset(wall=True)
                #self.env.obs_well_show()
                agent.train(obs)

                # gathering past trajectories
                for step in range(self.num_step):
                    action = agent.act(obs)
                    self.action_count[agent_index, action] += 1
                    spatial_concat_action = np.zeros((self.env.height, self.env.width, 5))
                    spatial_concat_action[:, :  action] = 1
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
                sr = np.zeros((11, 11, 3))
                sr[self.env.agent_xy[0], self.env.agent_xy[1], :] = 1
                gamma = np.array([0.5 , 0.9, 0.99])
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
            self.target_policy[agent_index] = agent.V.reshape((11, 11, 1))
            self.current_state[agent_index] = curr_obs
            self.target_action[agent_index] = target_action
            if consumed != None:
                self.target_preference[agent_index, consumed] = 1
            self.target_sr[agent_index] = sr
        #print('Past Traj : {} Curr state : {} Targ action : {} Targ goal : {} Targ SR : {} Targ Done : {}'.format(
        #    self.past_trajectories.shape, self.current_state.shape, self.target_action.shape, self.target_preference.shape,
        #    self.target_sr.shape, self.target_policy.shape, self.dones.shape
        #))
        return self.past_trajectories, self.current_state, self.target_action, self.target_preference, self.target_sr, \
               self.target_policy, self.dones

    def get_most_act(self):
        action_count = copy.deepcopy(self.action_count)
        action_count /= np.reshape(np.sum(action_count, axis=-1), (-1, 1))

        return np.argmax(action_count, axis=-1), np.max(action_count, axis=-1)







if __name__ == '__main__':
    from environment.env import GridWorldEnv
    import agent
    env_config = dict(height=11, width=11, pixel_per_grid=8, preference=100, exp=2, save=True)
    env = GridWorldEnv(env_config)
    agent_config = dict(name='reward_seeking', species=[0.01], num=1000)
    agent = agent.agent_type[agent_config['name']]
    print(env.num_wall)
    storage = Storage(env, population=[agent(alpha=0.01, num_action=5, move_penalty=-0.01)], num_past=1, num_step=1)