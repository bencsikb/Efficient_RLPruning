# actor-critic network for Reinforcement Learning
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torch.autograd import Variable

import argparse
import math
import os
import random
from torch.distributions import Categorical
from varname import nameof


from models.models import *
from utils.LR_utils import normalize, denormalize, get_state, get_state2, get_layers_forpruning, list2FloatTensor, test_alpha_seq
from models.LR_models import actorNet, criticNet, actorNet2, init_weights
from utils.RL_rewards import reward_function,  reward_function2,reward_function3, reward_function4, reward_function5, reward_function6
from models.error_pred_network import errorNet
from utils.LR_losses import CriticLoss, ActorLoss, ActorPPOLoss, get_discounted_reward, get_advantage, get_discounted_reward
#from utils.state_tester import get_fix_state
from utils.RL_logger import RLLogger, TensorboardLogger

torch.manual_seed(42)
torch.cuda.manual_seed(42)

timefile = "/home/blanka/YOLOv4_Pruning/sandbox/time_measure_pruning3.txt"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #parser.add_argument('--layers_for_pruning', default=[0, 2, 5, 11, 15, 19, 24, 28, 32, 35, 38, 41, 44, 47, 50, 55, 59, 63, 66, 69, 72, 75, 78, 81, 86, 90, 94, 97, 100, 105, 107, 115, 117, 123, 125, 127, 133, 135, 137, 144, 146, 155, 157, 159])
    parser.add_argument('--yolo_layers', default=[138, 148, 149, 160])
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--episodeNum', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--ent_coef', type=int, default=5e-3)
    parser.add_argument('--actor-base-lr', type=int, default=1e-3)
    parser.add_argument('--actor-last-lr', type=int, default=5e-8)
    parser.add_argument('--critic-base-lr', type=int, default=0.01)
    parser.add_argument('--test-case', type=str, default='repr_58_c_2')
    parser.add_argument('--save-interval', type=int, default=50)
    #parser.add_argument('--ppo-eps-base', type=int, default=4)
    #parser.add_argument('--ppo-eps-last', type=int, default=4)


    # For reward function
    parser.add_argument('--reward-func', type=str, default="reward_function")
    parser.add_argument('--err_coef', type=int, default=1.1)
    parser.add_argument('--spars_coef', type=int, default=1)
    parser.add_argument('--target_error', type=int, default=0.2)
    parser.add_argument('--target_spars', type=int, default=0.6)
    parser.add_argument('--beta', type=int, default=5)
    #parser.add_argument('--A', type=int, default=2)
    #parser.add_argument('--B', type=int, default=1)
    #parser.add_argument('--N', type=int, default=2)

    # Flags
    parser.add_argument('--variable_logflag', type=bool, default=True)
    parser.add_argument('--lr_sched_step_flag', type=bool, default=False)
    parser.add_argument('--set_new_lr', type=bool, default=True)
    parser.add_argument('--PPO-flag', type=bool, default=False)
    parser.add_argument('--set-new-lossfunc', type=bool, default=False)
    parser.add_argument('--test', type=bool, default=False)

    # Pretrained nets
    parser.add_argument('--network_forpruning', type=str, default='/data/blanka/runs/exp_kitti/weights/last.pt')
    parser.add_argument('--cfg', type=str, default='cfg/yolov4_kitti.cfg', help='model.yaml path')
    parser.add_argument('--error_pred_network', type=str, default='/data/blanka/checkpoints/pruning_error_pred/test_97_2534.pth')
    parser.add_argument('--pretrained', type=str, default='')

    # Destinations
    parser.add_argument('--ckpt-save-path', type=str, default='checkpoints/ReinforcementLearning')
    parser.add_argument('--results-save-path', type=str, default='results/ReinforcementLearning')
    parser.add_argument('--log-dir', type=str, default='/home/blanka/YOLOv4_Pruning/logs')
    parser.add_argument('--tb-log-dir', type=str, default='/home/blanka/YOLOv4_Pruning/runs/ReinforcementLearning', help="Tensorboard logging directoty")

    #parser.add_argument('--logdir', type=str, default='runs/pruning_error', help='tensorboard log path')

    opt = parser.parse_args()

    torch.autograd.set_detect_anomaly(True)
    # Create logger
    rl_logger = RLLogger(log_dir=opt.log_dir, test_case=opt.test_case)
    tb_logger = TensorboardLogger(log_dir=opt.tb_log_dir, test_case=opt.test_case)

    # Load pretrained nets
    #opt.pretrained = "/home/blanka/YOLOv4_Pruning/checkpoints/ReinforcementLearning/test_final_58_c_1D_250.pth"
    net_for_pruning = Darknet(opt.cfg).to(opt.device)
    ckpt_nfp = torch.load(opt.network_forpruning)
    state_dict = {k: v for k, v in ckpt_nfp['model'].items() if net_for_pruning.state_dict()[k].numel() == v.numel()}
    net_for_pruning.load_state_dict(state_dict, strict=False)
    network_size = len(net_for_pruning.module_list)

    # Load pretrained error prediction net
    ckpt_epn = torch.load(opt.error_pred_network)
    error_pred_net = ckpt_epn['model']
    error_pred_net.eval()

    # Initialize actor and critic networks

    if opt.pretrained:
        ckpt = torch.load(opt.pretrained)
        actorNet = ckpt['actor_model']
        print(actorNet)
        criticNet = ckpt['critic_model']
        actor_optimizer = ckpt['actor_optimizer']
        critic_optimizer = ckpt['critic_optimizer']
        episode = ckpt['episode']
        lr_sched = ckpt['lr_sched']
        log_probs_prev = ckpt['log_probs']

        # Change learning rate
        if opt.set_new_lr:
            for g in actor_optimizer.param_groups:
                g['lr'] = opt.actor_base_lr
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=opt.episodeNum, eta_min=opt.actor_last_lr,
                                                                  last_epoch=episode)
            lr_sched.step()

        critic_criterion = ckpt['critic_criterion']
        actor_criterion = ActorPPOLoss().to(opt.device) if opt.set_new_lossfunc else ckpt['actor_criterion']
        print("pretrained", episode)
        eps = 3.964
    else:
        print("new model")

        actorNet = actorNet2(44, 23).to(opt.device)
        #actorNet.apply(init_weights)
        criticNet = criticNet(44, 1).to(opt.device)
        actor_optimizer = torch.optim.Adam(actorNet.parameters(), lr=opt.actor_base_lr)
        lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(actor_optimizer, T_max=opt.episodeNum, eta_min=opt.actor_last_lr, last_epoch=-1)
        critic_optimizer = torch.optim.Adam(criticNet.parameters(), lr=opt.critic_base_lr)
        # Define loss functions
        critic_criterion = CriticLoss().to(opt.device)
        actor_criterion = ActorPPOLoss().to(opt.device) if opt.PPO_flag else ActorLoss().to(opt.device)
        episode = 0


    # Get layer indexes that can be pruned
    layers_for_pruning = get_layers_forpruning(net_for_pruning, opt.yolo_layers)

    # Define alpha values
    alphas = np.arange(0.0, 2.3, 0.1).tolist()
    alphas = [float("{:.2f}".format(x)) for x in alphas]

    # Log settings
    settings_dict = { "episode": episode,
                      "actor_lr": actor_optimizer.param_groups[0]['lr'],
                      "critic_lr": critic_optimizer.param_groups[0]['lr'],
                      "actor_lr_sched": lr_sched.get_lr()[0]
    }
    rl_logger.log_settings(opt, settings_dict)



    while episode < opt.episodeNum:
        start_time_episode = time.time()

        network_seq = []
        # Deepcopy of the initialized network
        #for ni in range(opt.batch_size):
        #    network_seq.append(pickle.loads(pickle.dumps(net_for_pruning)))
        init_param_nmb = sum([param.nelement() for param in net_for_pruning.parameters()])

        action_seq = torch.full([opt.batch_size, 1, 44], -1.0)
        state_seq = torch.full([opt.batch_size, 6, 44], -1.0)

        actions = []
        states = []
        rewards = []
        rewards_list = []
        values = []
        policies = []
        log_probs = []
        entropies = []
        errors = []
        dEs, dSs = [], []

        layer_cnt = 0
        sparsity_prev = torch.full([opt.batch_size], -1.0)
        error_prev = torch.full([opt.batch_size], -1.0)


        for layer_i in range(network_size):

            if layer_i in layers_for_pruning:
                #print("Pruning layer ", layer_cnt, layer_i)
                sequential_size = len(net_for_pruning.module_list[layer_i])
                layer = [net_for_pruning.module_list[layer_i][j] for j in range(sequential_size) if isinstance(net_for_pruning.module_list[layer_i][j], nn.Conv2d)]
                layer = layer[0]

                # Get state
                #state_seq = Variable(get_state2(state_seq, sparsity_prev, layer, layer_cnt), requires_grad=True)
                state_seq = get_state2(state_seq, sparsity_prev, layer, layer_cnt)

                data = state_seq[:,-1, :].view([opt.batch_size, -1]).type(torch.float32).to(opt.device)
                #data = state_seq.view([opt.batch_size, -1]).type(torch.float32).to(opt.device)


                #print("data req grad", layer_cnt, state_seq.requires_grad)

                probs, action_dist, log_softmax = actorNet(data)
                q_value = criticNet(data)
                #print("dist", action_dist)
                #print("value", q_value)

                # Choose alpha values based on probabilities
                action = action_dist.sample() # alpha index
                #entropy = action_dist.entropy()
                log_prob = action_dist.log_prob(action).unsqueeze(1)
                policy = probs.gather(-1, action.unsqueeze(0))
                entropy = - (probs * log_softmax).sum(1, keepdim=True)
                if layer_cnt == 0 or layer_cnt == 43:
                    rl_logger.log_probs(probs, episode, layer_cnt)
                tb_logger.log_probs(probs, episode, layer_cnt)

                if layer_cnt == 0 or layer_cnt == 43:
                    tb_logger.log_probs_merged(probs, episode, layer_cnt)

                for i in range(opt.batch_size):
                    action_seq[i, :, layer_cnt] = normalize(alphas[action[i]], 0.0, 2.2)


                # Get the error for every sample in the batch
                errorNet_input_data = torch.cat((action_seq, state_seq[:,-1, :].unsqueeze(1)), dim=1).view([opt.batch_size, -1]).type(torch.float32).to(opt.device)
                prediction = error_pred_net(errorNet_input_data)
                #if errorNet_input_data[0,40] == -1.0 and errorNet_input_data[0,0] and errorNet_input_data[0,20] == -1.0:
                error, sparsity = prediction[:,0], prediction[:,1]
                errors.append(error.unsqueeze(1))

                #reward = reward_function(A=denormalize(error, 0, 1), Ta=opt.target_error, S=denormalize(sparsity, 0, 1), Ts=opt.target_spars, device=opt.device)
                #reward = reward_function3(E=denormalize(error-error_prev, 0, 1), S=denormalize(sparsity-sparsity_prev, 0, 1), Te=opt.target_error, Ts=opt.target_spars, saprs_coef=opt.spars_coef, err_coef=opt.err_coef, device=opt.device)
                #dE = torch.sub(denormalize(error,0,1), denormalize(error_prev.to(opt.device),0,1))
                #dS = torch.sub(denormalize(sparsity,0,1), denormalize(sparsity_prev.to(opt.device),0,1))
                #reward = eval(opt.reward_func)( error, sparsity, opt.spars_coef, opt.err_coef, opt.device) # reward_functio4
                #reward = eval(opt.reward_func)( denormalize(error, 0, 1), denormalize(sparsity, 0, 1), opt.target_spars, opt.beta, opt.device) # reward_functio5
                reward = eval(opt.reward_func)(denormalize(error, 0, 1), opt.target_error, denormalize(sparsity, 0, 1), opt.target_spars, opt.err_coef, opt.spars_coef, opt.device, opt.beta ) # reward_function
                #reward = eval(opt.reward_func)( error, sparsity, opt.A, opt.B, opt.N, opt.device) # reward_functio6


                reward = reward.unsqueeze(1)
                rewards_list.append(reward)
                #print("rewarrrd ", reward)

                #if prediction[0,0]<0:7
                if False:
                    print(errorNet_input_data[0])
                    print("prediction", prediction[0])
                    print("reward ", reward[0])
                    print(log_prob[0])

                #if layer_cnt == 43:
                #    print("inal_spars", sparsity)
                #    print("final reward ", reward)

                layer_cnt += 1
                sparsity_prev = sparsity.clone()
                error_prev = error.clone()

                # Save the trajectory
                #log_probs.append(log_prob.clone().detach())
                log_probs.append(log_prob)

                entropies.append(entropy)
                list2FloatTensor(entropies)
                actions.append(action_seq.clone().detach())
                states.append(state_seq.clone())
                #rewards.append(denormalize(reward, 0, 1))
                #values.append(denormalize(q_value, 0, 1))
                rewards.append(reward)
                values.append(q_value)
                policies.append(policy)
                #dEs.append(dE.clone().detach().unsqueeze(1))
                #dSs.append(dS.clone().detach().unsqueeze(1))


        # Test the results
        if opt.test:
            test_alpha_seq(denormalize(list2FloatTensor(errors)[-1,:,0], 0, -1),
                           denormalize(states[-1][:, -1, -1], 0, 1),
                           list2FloatTensor(rewards_list)[-1, :, 0],
                           denormalize(actions[-1][:, 0, :], 0, 2.2),
                           opt.yolo_layers,
                           test_case="test_58_d_2700",
                           error_thresh=None, spars_thresh=None, reward_thresh=-5)
            break

        returns  = get_discounted_reward(list2FloatTensor(rewards),  list2FloatTensor(values), gamma=0.99)

        if episode==0:
            print(episode, "notlenlogprob")
            log_probs_prev = torch.zeros(list2FloatTensor(log_probs).shape)
        else:
            #print(len(log_probs_prev), log_probs_prev[0].shape)
            #print(len(log_probs), log_probs[0].shape)
            log_probs_prev = list2FloatTensor(log_probs_prev).detach()


        #print("prev", log_probs_prev[0])
        #print("cur", log_probs[0])

        # Loss backwards here
        critic_loss = critic_criterion(list2FloatTensor(rewards),  list2FloatTensor(values), 0.99)
        #critic_loss.backward(retain_graph=True)
        if opt.PPO_flag:
            eps = opt.ppo_eps_base - episode*(opt.ppo_eps_base-opt.ppo_eps_last)/opt.episodeNum
            actor_loss = actor_criterion(list2FloatTensor(rewards), list2FloatTensor(values), list2FloatTensor(policies), list2FloatTensor(log_probs), log_probs_prev, list2FloatTensor(entropies), ent_coef=opt.ent_coef, gamma=0.99, eps=eps)
        else:
            actor_loss = actor_criterion(list2FloatTensor(rewards),  list2FloatTensor(values), list2FloatTensor(policies), list2FloatTensor(log_probs), list2FloatTensor(entropies), ent_coef=opt.ent_coef, gamma=0.99)
        log_probs_prev = log_probs

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()

        final_loss = actor_loss + critic_loss
        #final_loss.backward()
        final_loss.backward(retain_graph=True)
        reward_backprop = list2FloatTensor(rewards).mean()
        (-reward_backprop).backward()

        actor_optimizer.step()
        critic_optimizer.step()
        if opt.lr_sched_step_flag: lr_sched.step()


        print(critic_loss)
        print(actor_loss)

        # LOGGING

        if opt.variable_logflag:
            # print(actions[0])
            rl_logger.log_variables(denormalize(actions[-1][:,0,:], 0, 2.2), nameof(actions), episode=None)
            rl_logger.log_variables(list2FloatTensor(values), nameof(values), episode=None)
            rl_logger.log_variables(denormalize(list2FloatTensor(errors)[:,:,0], 0, 1), nameof(error), episode=None)
            rl_logger.log_variables(denormalize(states[-1][:,-1,:], 0, 1), nameof(sparsity), episode=None)
            rl_logger.log_variables(list2FloatTensor(rewards_list)[:,:,0], nameof(rewards_list), episode=None)
            rl_logger.log_variables(returns, nameof(returns), episode=None)

            tb_logger.log_variables(episode, denormalize(states[-1][:,-1,:], 0, 1), nameof(sparsity), dim=0)
            tb_logger.log_variables(episode, denormalize(actions[-1][:,0,:], 0, 2.2), nameof(actions), dim=0)
            tb_logger.log_variables(episode, denormalize(list2FloatTensor(errors)[:,:,0], 0, 1), nameof(error), dim=1)
            tb_logger.log_variables(episode, list2FloatTensor(rewards_list)[:,:,0], nameof(rewards_list), dim=1)
            if opt.PPO_flag: tb_logger.log_scalar(episode, eps, nameof(eps))

            #tb_logger.log_variables(episode, list2FloatTensor(dEs)[:,:,0], nameof(dE), dim=1)
            #tb_logger.log_variables(episode, list2FloatTensor(dSs)[:,:,0], nameof(dS), dim=1)


        # Save parameters
        rl_logger.log_results(opt.results_save_path, episode, critic_loss.item(), actor_loss.item(), list2FloatTensor(rewards).mean().item())
        rl_logger.log_learning_rate(episode, lr_sched)
        tb_logger.log_results(episode, critic_loss.item(), actor_loss.item(), list2FloatTensor(rewards).mean().item())
        tb_logger.log_learning_rate(episode, lr_sched)
        # Save checkpoint
        checkpoint = { 'episode': episode,
                       'actor_model': actorNet,
                       'critic_model': criticNet,
                       'actor_optimizer': actor_optimizer,
                       'critic_optimizer': critic_optimizer,
                       'actor_criterion': actor_criterion,
                       'critic_criterion': critic_criterion,
                       'lr_sched': lr_sched,
                       'log_probs': log_probs_prev
        }
        if opt.PPO_flag: checkpoint['eps'] = eps
        ckp_save_path = os.path.join(opt.ckpt_save_path, opt.test_case)
        torch.save(checkpoint, ckp_save_path + ".pth")
        # Save the checkpoint with episode
        if episode % opt.save_interval == 0:
            ckp_save_path = os.path.join(opt.ckpt_save_path, opt.test_case + "_" + str(episode))
            torch.save(checkpoint, ckp_save_path + ".pth")



        episode += 1
        print("Episode time ", time.time() - start_time_episode)





















