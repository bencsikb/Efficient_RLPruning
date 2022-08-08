import os
import argparse
import json
import torch
import numpy as np
from varname import nameof
from torch.utils.tensorboard import SummaryWriter


class RLLogger():

    def __init__(self, log_dir, test_case):
        super().__init__()

        self.log_dir = os.path.join(log_dir, test_case)
        self.test_case = test_case
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)

    def log_settings(self, opt, settings_dict: dict):
        path = os.path.join(self.log_dir, "settings.txt")
        with open(path, 'a+') as file:
            json.dump(opt.__dict__, file, indent=2)
            file.write("\n")
            file.write(json.dumps(settings_dict))

    def log_results(self, results_dir, episode, critic_loss, actor_loss, reward):
        path = os.path.join(results_dir, self.test_case)
        with open(path, 'a+') as f:
            f.write(str(episode) + " " + str(critic_loss) + " " + str(actor_loss) + " " + str(reward) + "\n")

    def log_learning_rate(self, episode, lr_sched):
        learning_rate = lr_sched.get_lr()[0]
        path = os.path.join(self.log_dir, "learning_rate.txt")
        with open(path, 'a+') as pf:
            pf.write(str(episode) + " " + str(learning_rate) + "\n")


    def log_probs(self, probs, episode, layer_cnt):
        path = os.path.join(self.log_dir, "probs.txt")
        with open(path, 'a+') as pf:
            pf.write(str(episode) + " " + str(layer_cnt) + " " + str(probs[0]) + "\n")
            if layer_cnt == 43: pf.write("\n")

    def log_variables(self, variable, name, episode=None):

        path = os.path.join(self.log_dir, name + ".txt")
        #if episode in None:
        #    path = os.path.join(self.logdir, name + ".txt")
        #else:
        #    path = os.path.join(self.logdir, name + "_" +  str(episode) + ".txt")

        with open(path, "w") as f:
            if torch.is_tensor(variable):
                f.write(str(variable.shape) + "\n")
            elif len(variable):
                f.write(str(len(variable)) + "\n")
                if torch.is_tensor(variable[0]):
                    f.write(str(variable[0].shape) + "\n")

            f.write(str(variable) + "\n")

class TensorboardLogger():
    def __init__(self, log_dir, test_case):
        super().__init__()

        self.log_dir = os.path.join(log_dir, test_case)
        self.test_case = test_case
        if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)

    def log_results(self, episode, critic_loss, actor_loss, reward):
        tags = ['results/critic_loss', 'results/actor_loss', 'results/rewards']
        self.tb_writer.add_scalar(tags[0], critic_loss, episode)
        self.tb_writer.add_scalar(tags[1], actor_loss, episode)
        self.tb_writer.add_scalar(tags[2], reward, episode)

    def log_probs(self, probs, episode, layer_cnt):
        """
        Logs the probabilities predicted by the actor for the first instance in the batch
        in each episode, each layer.
        """
        alphas = np.arange(0.0, 2.3, 0.1).tolist()
        alphas = [float("{:.2f}".format(x)) for x in alphas]

        for i, prob in enumerate(probs[0]):
            tag = F"layer_{layer_cnt}/alpha_{alphas[i]}"
            self.tb_writer.add_scalar(tag, prob, episode)

    def log_probs_merged(self, probs, episode, layer_cnt):
        """
        Logs the probabilities of 0.0 and 2.2 alpha value on a same graph for each layer in the
        first sample in the batch.
        """
        alphas = np.arange(0.0, 2.3, 0.1).tolist()
        alphas = [float("{:.2f}".format(x)) for x in alphas]

        tag00 = F"merged/alpha_{alphas[0]}"
        tag22 = F"merged/alpha_{alphas[-1]}"
        prob00 = probs[0][0]
        prob22 = probs[0][-1]
        self.tb_writer.add_scalars(tag00, {f"layer_{layer_cnt}": prob00}, episode)
        self.tb_writer.add_scalars(tag22, {f"layer_{layer_cnt}": prob22}, episode)

    def log_learning_rate(self, episode, lr_sched):

        tag = 'learning_rate'
        lr = lr_sched.get_lr()[0]
        self.tb_writer.add_scalar(tag, lr, episode)

    def log_variables(self, episode,  variable, name, dim):

        # [32, 1, 44] --> [1,44]
        avgs = torch.mean(variable, dim)
        stds = torch.std(variable, dim)

        for i, (avg, std) in enumerate(zip(avgs, stds)):
            tag = F"{name}/{nameof(avg)}/layer_{i}"
            self.tb_writer.add_scalar(tag, avg, episode)
            tag = F"{name}/{nameof(std)}/layer_{i}"
            self.tb_writer.add_scalar(tag, std, episode)

    def log_scalar(self, episode, scalar, name):
        tag = name
        self.tb_writer.add_scalar(tag, scalar, episode)



