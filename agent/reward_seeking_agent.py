# from agent.agent import BaseAgent
import sys

sys.path.append('./')
import numpy as np
from environment.env import GridWorldEnv



# from agent.agent import BaseAgent
import numpy as np
from environment.env import GridWorldEnv

# class RewardSeekingAgent(BaseAgent):
class RewardSeekingAgent():
    '''
    objects : ['Blue', 'Pink', 'Orange', 'Green']
    action : ['Stay', 'Down', 'Right', 'Up', 'Left']
    '''

    def __init__(self, alpha, num_action=5, move_penalty=-0.01):
        self.alpha = alpha
        self.num_action = num_action
        self.num_object = 4
        self.reward = np.random.dirichlet(np.full((self.num_object, ), self.alpha))
        self.discount_factor = 1
        self.theta = 0.0001
        self.move_penalty = move_penalty
        self.most_prefer = np.argmax(self.reward)
        self.wall_reward = -0.05

    def act(self, observation):
        agent_channel = observation[:, :, -1]
        agent_xs, agent_ys = np.where(agent_channel == 1)
        agent_x, agent_y = agent_xs[0], agent_ys[0]
        action = -1
        for i in range(self.num_action):
            if self.policy[agent_x, agent_y, i] == 1:
                action = i
                break
        if action == -1:
            raise ValueError("value iteration error..!")
        return action

    def train(self, observation):
        self.obs_height = observation.shape[0]
        self.obs_width = observation.shape[1]

        P = {}
        grid = np.arange(self.obs_height * self.obs_width).reshape([self.obs_height, self.obs_width])
        it = np.nditer(grid, flags=['multi_index'])
        trial = 0

        while not it.finished:
            state = it.iterindex
            x, y = it.multi_index

            # P[x][y][a] = (prob, next_state, reward, done)
            P[state] = {a: [] for a in range(self.num_action)}

            done, reward = self._done_reward(observation, x, y)
            prob=1

            if done:
                P[state][0] = [(prob, [x, y], reward, True)]
                P[state][1] = [(prob, [x, y], reward, True)]
                P[state][2] = [(prob, [x, y], reward, True)]
                P[state][3] = [(prob, [x, y], reward, True)]
                P[state][4] = [(prob, [x, y], reward, True)]
            else:
                # action: ['Stay', 'Down', 'Right', 'Up', 'Left']
                next_s_0 = [x, y]
                next_s_1 = [x + 1, y] if (x + 1) < self.obs_height else [x, y]
                next_s_2 = [x, y + 1] if (y + 1) < self.obs_width else [x, y]
                next_s_3 = [x - 1, y] if (x - 1) > -1 else [x, y]
                next_s_4 = [x, y - 1] if (y - 1) > -1 else [x, y]

                P[state][0] = [(prob, next_s_0, reward, self._done_reward(observation, next_s_0[0], next_s_0[1])[0])]
                P[state][1] = [(prob, next_s_1, reward, self._done_reward(observation, next_s_1[0], next_s_1[1])[0])]
                P[state][2] = [(prob, next_s_2, reward, self._done_reward(observation, next_s_2[0], next_s_2[1])[0])]
                P[state][3] = [(prob, next_s_3, reward, self._done_reward(observation, next_s_3[0], next_s_3[1])[0])]
                P[state][4] = [(prob, next_s_4, reward, self._done_reward(observation, next_s_4[0], next_s_4[1])[0])]

            P[state][0] = [(1.0, [x, y], reward, done)]
            P[state][1] = [(1.0, [x+1, y], reward, done)]
            P[state][2] = [(1.0, [x, y+1], reward, done)]
            P[state][3] = [(1.0, [x-1, y], reward, done)]
            P[state][4] = [(1.0, [x, y-1], reward, done)]

            it.iternext()
            trial += 1
        policy, v = self._value_iteration(P)

    def _done_reward(self, observation, xi, yi):
        done = False
        reward = self.move_penalty  # reward by wall and objects

        # wall trap the agent. consuming objects.
        for thing_idx in range(self.num_object + 1):
            if observation[xi, yi, thing_idx] == 1:
                reward = self.wall_reward if thing_idx == 0 else self.reward[thing_idx - 1]
                done = True
                break

        return done, reward

    def _one_step_lookahead(self, xi, yi, P):
        A = np.zeros(self.num_action)
        for a in range(self.num_action):
            for prob, next_state, reward, done in P[xi * self.obs_width + yi][a]:
                next_x, next_y = next_state
                if done:
                    A[a] += prob * reward
                else:
                    A[a] += prob * (reward + self.discount_factor * self.V[next_x][next_y])
        return A

    def _value_iteration(self, P):
        self.V = np.zeros([self.obs_height, self.obs_width])
        while True:
            delta = 0
            for xi in range(self.obs_height):
                for yi in range(self.obs_width):
                    A = self._one_step_lookahead(xi, yi, P)
                    best_action_value = np.max(A)
                    delta = max(delta, np.abs(best_action_value - self.V[xi][yi]))
                    self.V[xi][yi] = best_action_value
            if delta < self.theta:
                break

        self.policy = np.zeros([self.obs_height, self.obs_width, 5])
        for xi in range(self.obs_height):
            for yi in range(self.obs_width):
                A = self._one_step_lookahead(xi, yi, P)
                best_action = np.argmax(A)
                self.policy[xi, yi, best_action] = 1.0

        return self.policy, self.V


if __name__ == '__main__':
    config = dict(height=11, width=11, pixel_per_grid=8, num_wall=0, preference=100, exp=2, save=True)
    env = GridWorldEnv(config)
    print(env.preference)
    print(env.prefer_reward)
    obs = env.reset()
    print('###############################################################')
    print('############### Check for reward seeking agent  ###############')
    print('###############################################################')
    env.obs_well_show()

    agent = RewardSeekingAgent(alpha=0.01, num_action=5)
    print('################# Check for agent preference #################')
    print("agent's preference: ", agent.most_prefer)
    print("0 : B, 1 : P, 2 : O, 3 : G")
    agent.train(env.observation)
    print('############## Check for agent reward function ###############')
    print(agent.reward)
    print('################## Check for agent policy ####################')
    print(np.argmax(agent.policy, axis=2))
    print("action index to text = ['Stay', 'Down', 'Right', 'Up', 'Left']")
    print('############## Check for agent value function ################')
    print(np.around(agent.V.reshape([11, 11]), 2))

    print('#################### Check for agent act #####################')
    env.obs_well_show()
    action = agent.act(env.observation)
    print(action)
    print("action index to text = ['Stay', 'Down', 'Right', 'Up', 'Left']")
    obs, reward, done, _ = env.step(action)
    env.obs_well_show()






