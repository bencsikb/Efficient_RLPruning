import torch
import numpy as np

def reward_function(A, Ta, S, Ts, err_coeff, spars_coeff, device, beta=5):
    """
    :param A: accuracy deterioration (error)
    :param Ta: desired accuracy deterioration (maximal)
    :param S: sparsity (percent of pruned parameters
    :param Ts: desired sparsity
    :return: reward
    """
    baseline = 100
    baseline_tens = torch.full(A.shape, baseline, dtype=torch.float32)

    zerotens = torch.zeros(A.shape).to(device)
    #reward = baseline_tens - beta* (torch.max( (A-Ta)/(1-Ta), zerotens) + torch.max( 1 - S/Ts, zerotens))
    reward = - beta* (err_coeff*torch.max( (A-Ta)/(1-Ta), zerotens) + spars_coeff*torch.max( 1 - S/Ts, zerotens))

    #print("A", A)
    #print("S", A-Ta)
    #print(torch.max( (A-Ta)/(1-Ta), zerotens))
    return reward

def reward_function2(params, error, sparsity, baseline):


    params_tens = torch.full(error.shape, params, dtype=torch.float32)
    baseline_tens = torch.full(error.shape, baseline, dtype=torch.float32)

    reward = baseline_tens -  error * torch.log(params_tens - sparsity * params_tens)

    return reward

def reward_function3(E, S, Te, Ts, err_coef, spars_coef, device):

    zerotens = torch.zeros(E.shape).to(device)


    reward = spars_coef*S/Ts - err_coef*torch.max((E-Te)/(1-Te),zerotens)

    return reward

def reward_function4(E, S,  err_coef, spars_coef, device):

    zerotens = torch.zeros(E.shape).to(device)
    onetens = torch.ones(E.shape).to(device)
    minusonetens = torch.full(E.shape, -1.0, dtype=torch.float32).to(device)
    E_sat = torch.min(torch.max(E, minusonetens), onetens)
    S_sat = torch.min(torch.max(S, minusonetens), onetens)


    reward = -err_coef*torch.abs(torch.sin(E_sat)) - spars_coef*torch.abs(torch.sin(S_sat))

    return reward

def reward_function5(E, S,  Ts, beta, device):

    zerotens = torch.zeros(E.shape).to(device)
    onetens = torch.ones(E.shape).to(device)

    reward =  beta * (onetens - torch.max(E, zerotens)) * (S/Ts)

    return reward

def reward_function6(E, S, a, b, n, device):

    atens = torch.full(E.shape, a, dtype=torch.float32).to(device)
    btens = torch.full(E.shape, b, dtype=torch.float32).to(device)
    onetens = torch.ones(E.shape).to(device)
    minusonetens = torch.full(E.shape, -1.0, dtype=torch.float32).to(device)
    E_sat = torch.min(torch.max(E, minusonetens), onetens)
    S_sat = torch.min(torch.max(S, minusonetens), onetens)

    reward =  - atens * torch.pow(E_sat, 2*n) + btens + (- atens * torch.pow(S_sat, 2*n) + btens )

    return reward