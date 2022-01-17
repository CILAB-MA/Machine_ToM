import experiment1.experiment as experiment1
import experiment2.experiment as experiment2
import argparse
from utils import utils

EXPERIMENTS = [experiment1, experiment2] # experiment 3 will be update


def parse_args():
    parser = argparse.ArgumentParser('For ToM Passive Exp')
    parser.add_argument('--num_epoch', '-e', type=int, default=500)
    parser.add_argument('--main_exp', '-me', type=int, default=2)
    parser.add_argument('--sub_exp', '-se', type=int, default=1)
    parser.add_argument('--num_agent', '-na', type=int, default=500)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--alpha', '-a', type=float, default=0.01)
    parser.add_argument('--save_freq', '-s', type=int, default=10)
    parser.add_argument('--train_dir', default='./data/exp_1_sub_4_agent_500_id_60/train/', type=str)
    parser.add_argument('--eval_dir', default='./data/exp_1_sub_4_agent_500_id_60/eval/', type=str)
    parser.add_argument('--act_pri', '-ap', type=bool, default=True)
    args = parser.parse_args()
    return args


def main(args):
    experiment_folder = utils.make_folder()
    EXPERIMENTS[args.main_exp - 1].run_experiment(num_epoch=args.num_epoch, main_experiment=args.main_exp,
                                              sub_experiment=args.sub_exp, num_agent=args.num_agent,
                                              batch_size=args.batch_size, lr=args.lr,
                                              experiment_folder=experiment_folder,
                                              alpha=args.alpha, save_freq=args.save_freq,
                                              train_dir=args.train_dir, eval_dir=args.eval_dir, act_pri=args.act_pri)


if __name__ == '__main__':
    args = parse_args()
    main(args)