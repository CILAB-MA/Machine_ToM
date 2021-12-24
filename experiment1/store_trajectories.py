import numpy as np
import copy


class Storage(object):

    def __init__(self, env, population, num_past, step):
        self.env = env
        self.past_trajectories = np.zeros([len(population), num_past, step, env.height, env.width, 11])
        self.current_state = np.zeros([len(population), env.height, env.width, 6])
        self.target_action = np.zeros([len(population), 5])
        self.dones = np.zeros([len(population), num_past, step, 1])
        self.population = population
        self.num_past = num_past
        self.action_count = np.zeros([len(population), 5])

    def extract(self, custom_env=-100):
        for agent_index, agent in enumerate(self.population):

            for past_epi in range(self.num_past):
                if np.sum(custom_env) > 0:
                    obs = self.env.reset(custom=custom_env[agent_index])
                else:
                    obs = self.env.reset()

                # gathering past trajectories
                for step in range(self.env.epi_max_step):
                    action = agent.act(obs)
                    self.action_count[agent_index, action] += 1
                    spatial_concat_action = np.zeros((self.env.height, self.env.width, 5))
                    spatial_concat_action[:, :,  action] = 1

                    obs_concat = np.concatenate([obs, spatial_concat_action], axis=-1)
                    self.past_trajectories[agent_index, past_epi, step] = obs_concat
                    self.dones[agent_index, past_epi, step] = 1

                    obs, reward, done, _ = self.env.step(action)
                    if done:
                        # 0 = done
                        break

            # gathering current_state
            for _ in range(1):
                curr_obs = self.env.reset()
                target_action = agent.act(curr_obs)
                self.current_state[agent_index] = curr_obs
                self.target_action[agent_index, target_action] = 1
            print('Agent {} make complete!'.format(agent_index))

        return dict(episodes=self.past_trajectories,
                    curr_state=self.current_state,
                    target_action=self.target_action,
                    dones=self.dones)

    def reset(self):
        self.past_trajectories = np.zeros(self.past_trajectories.shape)
        self.current_state = np.zeros(self.current_state.shape)
        self.target_action = np.zeros(self.target_action.shape)
        self.dones = np.zeros(self.dones.shape)
        self.action_count = np.zeros(self.action_count.shape)

    def get_most_act(self):
        action_count = copy.deepcopy(self.action_count)
        action_count /= np.reshape(np.sum(action_count, axis=-1), (-1, 1))

        return np.argmax(action_count, axis=-1), np.max(action_count, axis=-1)

