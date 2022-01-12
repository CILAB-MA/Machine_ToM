import matplotlib.pyplot as plt
import os, copy
import numpy as np
import cv2
from sklearn.manifold import TSNE


def get_train_figure(measures, dicts, save_path, filename):
    plt.figure()
    palette = ['seagreen', 'royalblue', 'violet', 'darkorange', 'olivedrab', 'deepskyblue']
    plt.plot(measures[i], color=palette[i])
    plt.title(filename[:-4])
    plt.legend(loc=1)
    filename_str = '{}_'.format(filename)
    dkeys = dicts.keys()
    for d in dkeys:
        vals = dicts[d]
        filename_str += '{}_{}_'.format(d, vals)

    filepath = os.path.join(save_path, filename_str)
    plt.savefig(filepath)
    plt.clf()

def get_test_figure(measures, dicts, save_path, filename='test.jpg'):
    plt.figure()
    palette = ['seagreen', 'royalblue', 'violet', 'darkorange', 'olivedrab', 'deepskyblue']
    plt.plot(measures[i], color=palette[i])
    filename_str = '{}_'.format(filename)
    dkeys = dicts.keys()
    for d in dkeys:
        vals = dicts[d]
        filename_str += '{}_{}_'.format(d, vals)

    plt.legend(title='Test_Result', loc=1)
    filepath = os.path.join(save_path, filename_str)
    plt.savefig(filepath)
    plt.clf()

def visualize_embedding(e_char, labels, count, save_path, filename='e_char.jpg'):
    color_palette = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [204/255, 0, 204/255]])
    colors = color_palette[labels] * np.reshape(count, (-1, 1))
    plt.figure()
    plt.scatter(e_char[:, 0], e_char[:, 1], c=colors)
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath)
    plt.clf()


class Visualizer:

    def __init__(self, output_dir, grid_per_pixel, max_epoch, height, width):

        self.palette = [(0, 0, 0), (255, 0, 0), (255, 0, 255), (0, 128 ,255), (0, 255, 0), (177, 206, 251), [255, 255, 255]]
        self.output_dir = output_dir
        self.max_epoch = max_epoch
        self.grid_per_pixel = grid_per_pixel
        self.height = height
        self.width = width

    def _get_pt(self, x, y, action=0):
        left = (y * self.grid_per_pixel, x * self.grid_per_pixel + 4), (
            y * self.grid_per_pixel + 6, x * self.grid_per_pixel + 7), (
                 self.grid_per_pixel * y + 6, x * self.grid_per_pixel + 1)

        right = (y * self.grid_per_pixel + 2, x * self.grid_per_pixel + 7), (
            y * self.grid_per_pixel + 2, x * self.grid_per_pixel + 1), (
            self.grid_per_pixel * y + 8, self.grid_per_pixel * x + 4)

        up = (y * self.grid_per_pixel + 7, x * self.grid_per_pixel + 6), (
            y * self.grid_per_pixel + 1, x * self.grid_per_pixel + 6), (
                y * self.grid_per_pixel + 4, x * self.grid_per_pixel)

        down = (y * self.grid_per_pixel + 7, x * self.grid_per_pixel + 1), (
            y * self.grid_per_pixel + 1, x * self.grid_per_pixel + 1), (
                   y * self.grid_per_pixel + 4, x * self.grid_per_pixel + 7)
        actions = [0, down, right, up, left]
        if action == 0:
            return -100
        else:
            pt = actions[action]
        pt = np.array([pt])
        return pt

    def get_past_traj(self, obs, agent_xys, past_actions, epoch, foldername='past_traj', sample_num=0):

        palette = copy.deepcopy(self.palette) #black, #blue, #pink, #orange, #green, #skin, # white
        vis_obs = np.full((self.grid_per_pixel * self.height,
                           self.grid_per_pixel * self.width, 3),
                          255, dtype=np.uint8)

        for i in range(7):
            if i != 6:  # env(0, 1, 2, 3, 4), agent(5)
                xy_object = np.where(obs[:, :, i] == 1)
                xs, ys = xy_object
            else:
                ord, xs, ys = agent_xys

            for n , (x, y) in enumerate(zip(xs, ys)):
                if i == 6: #6 past_trajactory

                    clr = copy.deepcopy(palette[i])
                    clr[0] -= 255 / (ord[n] + 1)
                    clr[1] -= 255 / (ord[n] + 1)
                    pt = self._get_pt(x, y, past_actions[n])
                    if np.sum(pt) < 0:
                        continue
                    else:
                        cv2.drawContours(vis_obs, [pt], 0, clr, -1)
                    # vis_obs[x * self.grid_per_pixel: (x + 1) * self.grid_per_pixel,
                    # y * self.grid_per_pixel : (y + 1) * self.grid_per_pixel, :] = clr
                else:
                    vis_obs[x * self.grid_per_pixel: (x + 1) * self.grid_per_pixel,
                    y * self.grid_per_pixel : (y + 1) * self.grid_per_pixel, :] = palette[i]

        tozero = len(str(self.max_epoch)) - len(str(epoch))

        fn = tozero * '0' + str(epoch) + '.jpg'
        output_file = os.path.join(self.output_dir, str(sample_num), foldername, fn)

        if not os.path.exists(os.path.join(self.output_dir, str(sample_num), foldername)):
            os.makedirs(os.path.join(self.output_dir, str(sample_num), foldername))

        cv2.imwrite(output_file, vis_obs)


    def get_curr_state(self, obs, epoch, foldername='current_state', sample_num=0):
        palette = copy.deepcopy(self.palette)
        vis_obs = np.full((self.grid_per_pixel * self.height,
                           self.grid_per_pixel * self.width, 3),
                          255, dtype=np.uint8)
        for i in range(6):
            xy_object = np.where(obs[:, :, i] == 1)

            xs, ys = xy_object
            for x, y in zip(xs, ys):
                vis_obs[x * self.grid_per_pixel: (x + 1) * self.grid_per_pixel,
                y * self.grid_per_pixel: (y + 1) * self.grid_per_pixel, :] = palette[i]

        tozero = len(str(self.max_epoch)) - len(str(epoch))

        fn = tozero * '0' + str(epoch) + '.jpg'
        output_file = os.path.join(self.output_dir, str(sample_num), foldername, fn)

        if not os.path.exists(os.path.join(self.output_dir, str(sample_num), foldername)):
            os.makedirs(os.path.join(self.output_dir, str(sample_num), foldername))
        cv2.imwrite(output_file, vis_obs)

    def get_action(self, action_preds, epoch, foldername='action', sample_num=0):
        plt.figure()
        plt.ylabel('Prob')
        x = np.arange(5)
        plt.bar(x, np.exp(action_preds))
        plt.xticks(x, ['˚', '↓', '→', '↑', '←'])
        tozero = len(str(self.max_epoch)) - len(str(epoch))

        fn = tozero * '0' + str(epoch) + '.jpg'
        output_file = os.path.join(self.output_dir, str(sample_num), foldername, fn)

        if not os.path.exists(os.path.join(self.output_dir, str(sample_num), foldername)):
            os.makedirs(os.path.join(self.output_dir, str(sample_num), foldername))

        plt.savefig(output_file)
        plt.clf()

    def get_prefer(self, consumed_preds, epoch, foldername='consumed', sample_num=0):
        plt.figure()
        plt.ylabel('Prob')
        x = np.arange(5)
        plt.bar(x, np.exp(consumed_preds), color=['b', 'magenta', 'orange', 'g', 'black'])
        plt.xticks(x, ['Blue', 'Pink', 'Orange', 'Green', 'None'])

        tozero = len(str(self.max_epoch)) - len(str(epoch))


        fn = tozero * '0' + str(epoch) + '.jpg'
        output_file = os.path.join(self.output_dir, str(sample_num), foldername, fn)

        if not os.path.exists(os.path.join(self.output_dir, str(sample_num), foldername)):
            os.makedirs(os.path.join(self.output_dir, str(sample_num), foldername))

        plt.savefig(output_file)
        plt.clf()

    def get_sr(self, obs, sr_preds, epoch, foldername='sr', sample_num=0):
        palette = copy.deepcopy(self.palette)
        vis_obs = np.full((self.grid_per_pixel * self.height,
                           self.grid_per_pixel * self.width, 3),
                          255, dtype=np.uint8)
        for i in range(6, 0 , -1):
            if i != 6:
                xy_object = np.where(obs[:, :, i] == 1)
            else:
                xy_object = np.where(sr_preds[: , :, -1] != 0)

            xs, ys = xy_object
            for x, y in zip(xs, ys):
                if i == 6:
                    palette = copy.deepcopy(self.palette)
                    palette[i][0] -= 255 * sr_preds[x, y, -1]
                    palette[i][1] -= 255 * sr_preds[x, y, -1]
                vis_obs[x * self.grid_per_pixel: (x + 1) * self.grid_per_pixel,
                y * self.grid_per_pixel : (y + 1) * self.grid_per_pixel, :] = palette[i]

        tozero = len(str(self.max_epoch)) - len(str(epoch))

        fn = tozero * '0' + str(epoch) + '.jpg'
        output_file = os.path.join(self.output_dir, str(sample_num), foldername, fn)

        if not os.path.exists(os.path.join(self.output_dir, str(sample_num), foldername)):
            os.makedirs(os.path.join(self.output_dir, str(sample_num), foldername))

        cv2.imwrite(output_file, vis_obs)

    def get_action_char(self, e_char, most_act, count_act, epoch, foldername='action_e_char'):
        color_palette = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 1, 0], [204 / 255, 0, 204 / 255]])
        colors = color_palette[most_act] * np.reshape(count_act, (-1, 1))
        plt.figure()
        plt.scatter(e_char[:, 0], e_char[:, 1], c=colors)

        tozero = len(str(self.max_epoch)) - len(str(epoch))

        fn = tozero * '0' + str(epoch) + '.jpg'
        output_file = os.path.join(self.output_dir,  foldername, fn)

        if not os.path.exists(os.path.join(self.output_dir, foldername)):
            os.makedirs(os.path.join(self.output_dir, foldername))
        plt.savefig(output_file)
        plt.clf()

    def get_consume_char(self, e_char, preference, epoch, foldername='consume_e_char'):
        color_palette = ['blue', 'magenta', 'orange', 'limegreen', 'black']
        preference_index = np.argmax(preference, axis=-1)

        colors = [color_palette[i] for i in preference_index]
        plt.figure()
        plt.scatter(e_char[:, 0], e_char[:, 1], c=colors)


        tozero = len(str(self.max_epoch)) - len(str(epoch))

        fn = tozero * '0' + str(epoch) + '.jpg'
        output_file = os.path.join(self.output_dir, foldername, fn)

        if not os.path.exists(os.path.join(self.output_dir, foldername)):
            os.makedirs(os.path.join(self.output_dir, foldername))
        plt.savefig(output_file)
        plt.clf()


    def tsne_consume_char(self, e_char, preference, epoch, foldername='consume_e_char_tsne'):
        model = TSNE(2)
        tsne_results = model.fit_transform(e_char)

        color_palette = ['blue', 'magenta', 'orange', 'limegreen']
        preference_index = np.argmax(preference, axis=-1)

        al = np.max(preference, axis=-1)
        colors = [color_palette[i] for i in preference_index]
        plt.figure()
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], s=5, alpha=al, c=colors)

        tozero = len(str(self.max_epoch)) - len(str(epoch))

        fn = tozero * '0' + str(epoch) + '.jpg'
        output_file = os.path.join(self.output_dir, foldername, fn)

        if not os.path.exists(os.path.join(self.output_dir, foldername)):
            os.makedirs(os.path.join(self.output_dir, foldername))
        plt.savefig(output_file)
        plt.clf()