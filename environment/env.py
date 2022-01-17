import numpy  as np
import copy


class GridWorldEnv:
    def __init__(self, config):
        '''
        0 : Black(Wall)
        1 : Blue
        2 : Pink
        3 : Orange
        4 : Green
        5 : Agent
        '''
        self.observation_space = (config['height'], config['width'], 6)
        self.act_text = ['Stay', 'Down', 'Right', 'Up', 'Left']
        self.action_space = len(self.act_text)
        self.pixel_per_grid = config['pixel_per_grid']
        self.color_palette = {
            'BLUE': (0, 0, 255),
            'PINK': (255, 0, 255),
            'GREEN': (0, 255, 0),
            'ORANGE': (255, 128, 0),
            'WHITE': (255, 255, 255),
            'BLACK': (0, 0, 0)
            }
        self.preference = config['preference']
        self.prefer_reward = None
        self.epi_step = 0

        self.width = config['width']
        self.height = config['height']

        self.exp = config['exp']
        if self.exp == 1:
            self.epi_max_step = 1
            self.num_wall = 0
        elif self.exp == 2:
            self.epi_max_step = 31
            self.num_wall = np.random.randint(5)
        elif self.exp == 3 or self.exp == 4 or self.exp == 5:
            self.epi_max_step = 51
            self.num_wall = 6
        else:
            raise ValueError("Experiment index is not recognized")

    def reset(self, wall=False, custom=-100):
        self.epi_step = 0
        self.recent_action = None
        if np.sum(custom) > 0:
            self.observation = copy.deepcopy(custom)
            self.agent_xy = [np.where(self.observation[:, :, 5] == 1)[0], np.where(self.observation[:, :, 5] == 1)[1]]
            observation = copy.deepcopy(self.observation)
            return observation

        self.observation = np.full(self.observation_space, 0)
        # Make Base Wall
        goal_and_agent = self.observation.shape[-1] - 1  # except wall
        self.observation[0, :, 0] = 1
        self.observation[:, 0, 0] = 1
        self.observation[-1, :, 0] = 1
        self.observation[:, -1, 0] = 1

        # Place the additional walls if exist
        if wall:
            for _ in range(self.num_wall):
                direction = np.random.choice([0, 1] , 1)[0]
                st = np.random.randint(self.observation_space[0], size=2)
                if st[direction] == 10:
                    length = 1
                else:
                    length = np.random.randint(low=1, high=self.observation_space[0]-st[direction]+1)
                if direction == 0:
                    self.observation[st[0]:st[0]+length, st[1], 0] = 1
                else:
                    self.observation[st[0], st[1]:st[1]+length, 0] = 1

        avail_x = np.where(self.observation[:, :, 0] == 0)[0]
        avail_y = np.where(self.observation[:, :, 0] == 0)[1]
        rand_indices = np.random.choice(len(avail_x), goal_and_agent, replace=False)
        xs, ys = avail_x[rand_indices], avail_y[rand_indices]
        # Place the goals
        for i, (x, y) in enumerate(zip(xs, ys)):
            #print(x, y )
            if i + 1 == 5:
                self.agent_xy = [x, y]
            self.observation[x, y, i + 1] = 1
        observation = copy.deepcopy(self.observation)
        return observation

    def step(self, action):
        self.recent_action = action
        self.prev_xy = copy.deepcopy(self.agent_xy)
        _move = self._int_to_axis_move(action)
        self.observation[self.agent_xy[0], self.agent_xy[1], 5] = 0
        self.agent_xy[0] += _move[0]
        self.agent_xy[1] += _move[1]
        self.observation[self.agent_xy[0], self.agent_xy[1], 5] = 1
        self.epi_step += 1

        reward, done, consumed = self._check_done()
        info = dict(consumed=consumed)
        self.recent_reward = reward
        self.recent_done = done
        observation = copy.deepcopy(self.observation)
        return observation, reward, done, info

    def _check_done(self):
        '''
        The episode terminates if there is
        - collision with a wall : r = -1
        - consuming a goal
        - consuming preferred goal : r = 1
        - after 31 steps
        '''
        done = False
        reward = -0.01
        consumed = None
        if self.epi_step == self.epi_max_step:
            done = True
            reward = -0.01
        else:
            for i in range(self.observation_space[-1] - 1):
                if self.observation[self.agent_xy[0], self.agent_xy[1], i] == 1:
                    done = True
                    if i != 0:
                        if i == self.preference:
                            reward += self.prefer_reward[i - 1]
                        else:
                            reward += self.prefer_reward[i - 1]
                        consumed = i - 1
                    else:
                        reward += -0.05

        return reward, done, consumed

    def _int_to_axis_move(self, action):
        if action == 0:
            return [0, 0]
        elif action == 1:
            return [1, 0]
        elif action == 2:
            return [0, 1]
        elif action == 3:
            return [-1, 0]
        elif action == 4:
            return [0, -1]
        else:
            assert ('Your action input format is wrong {}, please in [0, 4]'.format(action))

    def obs_well_show(self):
        divider = '-' * 15
        if self.recent_action != None:
            info_msg = 'Step : {}, Action : {}, Done : {}, Reward : {}'.format(
                self.epi_step, self.act_text[self.recent_action], self.recent_done, self.recent_reward)
        else:
            info_msg = 'Step : 0, Action : None, Done : False, Reward : 0'
        obs = copy.deepcopy(self.observation)
        obs *= np.arange(1, self.observation_space[-1] + 1)
        checker = np.max(obs, axis=-1)
        checker2str = np.array2string(checker)
        checker2str = checker2str.replace('1', 'W').replace('2', 'B').replace('0', ' ')
        checker2str = checker2str.replace('3', 'P').replace('4','O').replace('5', 'G').replace('6', 'A')
        checker2str = checker2str.replace('[', ' ').replace(']', ' ')
        msg = divider + '\n' + info_msg + '\n' + checker2str + '\n' + divider
        print(msg)


if __name__ == '__main__':
    config = dict(height=25, width=25, pixel_per_grid=8, num_wall=1, preference=1, prefer_reward=[0, 0, 0, 1], exp=4, save=True)
    env = GridWorldEnv(config)
    env.prefer_reward = [0, 1, 0, 0]
    env.most_prefer = 1

    print('###############################################################')
    print('############## Check for env reset ###############')
    print('###############################################################')
    for _ in range(5):
        obs = env.reset()
        env.obs_well_show()
    obs = env.reset()
    print('###############################################################')
    print('############## Check for env and agent movement ###############')
    print('###############################################################')
    env.obs_well_show()

    # check step well
    for i in range(5):
        obs, rew, done, _ = env.step(i)
        env.obs_well_show()
        if done:
            break

    # check wall collision
    map = np.full((25, 25, 6), 0)
    map[0, :, 0] = 1
    map[:, 0, 0] = 1
    map[-1, :, 0] = 1
    map[:, -1, 0] = 1
    # agent xy
    map[1, 1, 5] = 1
    # prefer xy
    map[1, 2, 1] = 1
    # agent
    print('#################################################################')
    print('############## New custom map for checking reward ###############')
    print('#################################################################')
    env.reset(custom=map)
    env.obs_well_show()
    print('############## Check for Goal Reward ###############')
    obs, rew, done, _ = env.step(2)
    env.obs_well_show()

    env.reset(custom=map)
    print('##############Check for Collison Reward###############')
    obs, rew, done, _ = env.step(4)
    env.obs_well_show()

    print('##############Check for Making Wall###############')
    for i in range(1, 5):
        env.num_wall = i
        env.reset(wall=True)
        env.obs_well_show()


    print('##############Check for Done Wall###############')
    map2 = copy.deepcopy(map)


    env.reset(custom=map2)
    env.obs_well_show()
    obs, _, _, _ , =  env.step(3)
    env.obs_well_show()