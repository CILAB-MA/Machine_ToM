from agent.unreal import A3CNetwork
import torch as tr
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim

from environment.env import GridWorldEnv
from experiment3.config import get_configs
from utils.writer import Writer
from utils.dataset import Memory
import argparse, os

def parse_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()

class DeepRLAgent(BaseAgent):

    def __init__(self, num_action=5, device='cuda', is_train=False, agent_state='full',
                 model_path=None, save_path=None):

        if agent_state == 'full':
            pass
        elif agent_state == 'partial':
            pass
        elif agent_state == 'blind':
            pass

        self.model = A3CNetwork(num_input=6, num_action=num_action)
        self.device = device
        self.num_action = num_action
        self.optimizer = optim.Adam(self.agent.parameters(), lr=1e-5)
        if is_train:
            self.agent.train()
            self.model.train()
        else:
            self._load_model(model_path)
            self.model.eval()
        self.save_path = save_path
        self.entropy_coef = 5e-3

    def act(self, obs):
        obs = tr.from_numpy(obs).to(self.device)
        pi, v = self.model(obs)
        dist = Categorical(pi)
        action = dist.sample()
        return action.item(), pi

    def _load_model(self, model_path):
        self.model.load_state_dict(tr.load(model_path))

    def save_model(self):
        tr.save(self.model.state_dict(), self.save_path)

    def train(self, memory):
        batch = memory.get_batch()
        obss, acts, rews, next_obss, pis, dones = batch
        value_criterion = nn.MSELoss()
        obss = tr.from_numpy(obss, dtype=tr.float32).to(self.device)
        rews = tr.from_numpy(rews, dtype=tr.float32).to(self.device)
        next_obss = tr.from_numpy(next_obss, dtype=tr.float32).to(self.device)
        pis = tr.from_numpy(pis, dtype=tr.float32).to(self.device)
        dones = tr.from_numpy(dones, dtype=tr.float32).to(self.device)

        pred_pis, v = self.model(obss)
        _, next_v = self.model(next_obss)
        q = rews + self.discount * (1 - dones) * next_v
        a = q - v
        entropies =  -(pis.log() * pis).sum(1, keepdim=True)
        value_loss = value_criterion(v, q.detach())
        actor_loss = 0
        for i, pi in enumerate(pis):
            actor_loss += -a.detach() * pi.log()
        actor_loss /= len(pis)

        self.optimizer.zero_grad()
        (actor_loss + value_loss - self.entropy_coef * entropies).backward()
        self.optimizer.step()


def main(args):
    exp_kwargs, env_kwargs, model_kwargs, agent_kwargs = get_configs(args.num_exp)
    env = GridWorldEnv(**env_kwargs) # TODO : UPDATE THE ENVIRONMENT
    global_step, num_episode = 0, 0
    agent_kwargs['is_train'] = True
    agent = DeepRLAgent(**agent_kwargs)
    done = False
    writer = Writer(experiment_folder=args.lod_path)
    memory = Memory(args.batch_size, args.num_input)
    while global_step < 1e9:
        if done or global_step == 0:
            obs = env.reset()
            writer.write(os.path.join(args.base_dir), 'logs')
            if num_episode % args.save_freq == 0:
                agent.save_model()
            num_episode += 1
            memory.reset()
        action, pi = agent.act(obs)
        next_obs, rew, done, _ = env.step(action)
        memory.add(obs, action, rew, next_obs, pi[action].cpu().numpy(), done)

        if len(memory) > args.batch_size:
            agent.train(memory)

if __name__ == '__main__':
    args = parse_args() # TODO : UPDATE THE ARGUMENT PARSER
    main(args)






