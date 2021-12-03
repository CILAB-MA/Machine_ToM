import experiment1.experiment as experiment1
import experiment2.experiment as experiment2
import argparse
from utils import utils

EXPERIMENTS = [experiment1, experiment2] # experiment 3 will be update


def parse_args():
    parser = argparse.ArgumentParser('For ToM Passive Exp')
    parser.add_argument('--num_epoch', '-e', type=int, default=100)
    parser.add_argument('--main_exp', '-me', type=int, default=0)
    parser.add_argument('--sub_exp', '-se', type=int, default=1)
    parser.add_argument('--batch_size', '-b', type=int, default=16)
    parser.add_argument('--lr', '-l', type=float, default=1e-4)
    parser.add_argument('--alpha', '-a', type=float, default=0.01)
    parser.add_argument('--save_freq', '-s', type=int, default=10)
    args = parser.parse_args()
    return args


def main(args):
    experiment_folder = utils.make_folder()
    EXPERIMENTS[args.main_exp].run_experiment(num_epoch=args.num_epoch, main_experiment=args.main_exp,
                                              sub_experiment=args.sub_exp, batch_size=args.batch_size,
                                              lr=args.lr, experiment_folder=experiment_folder,
                                              alpha=args.alpha, save_freq=args.save_freq)


if __name__ == '__main__':
    args = parse_args()
    main(args)