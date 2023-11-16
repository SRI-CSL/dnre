import os
import argparse
import sys

import sbibm
import hamiltorch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
import uuid

from lampe.data import JointLoader, JointDataset, H5Dataset
from lampe.diagnostics import expected_coverage_ni
from lampe.inference import NRE, NRELoss, BNRELoss
from dnre.inference import *
from lampe.plots import nice_rc, corner, mark_point
from lampe.utils import GDStep
from sklearn.model_selection import ParameterGrid
import numpy as np

from joblib import dump, load
import concurrent.futures
import shutil

def build_log_posterior(estimator, thetas, D_in, D_out, prior, device = 'cpu'):
    
    S = thetas.shape[0]
    def posterior(theta, x):
        with torch.no_grad():
            if len(theta.shape) == 1:
                theta_repeat = theta[None].repeat(S,1)
                x_repeat = x[None].view(1,-1).repeat(S,1)
                thetas_repeat = thetas.clone().to(device)
            else:       
                theta_repeat = theta[None].repeat(S,1,1)
                x_repeat = x[None].view(1,1,-1).repeat(S,theta.shape[0],1)
                thetas_repeat = thetas[:,None].repeat(1, theta.shape[0], 1).to(device)
            
            log_r_inv = - estimator(theta_repeat.view(-1,D_in), thetas_repeat.view(-1,D_in), x_repeat.view(-1,D_out)).view(S,-1)
            p_theta = -(torch.logsumexp(log_r_inv, dim=0) - np.log(S)) + prior.log_prob(theta.cpu()).to(device)
        return p_theta
    return posterior

@torch.no_grad()
def log_post_obs(estimator, prior, task, ratio = False, S = 100):
    estimator.eval()
    post = torch.zeros(10)
    for i in range(1,11):
        observation = task.get_observation(num_observation=i)  # 10 per task
        x_star = observation.flatten()
        theta_true = task.get_true_parameters(i)
        if ratio:
            thetas = prior.sample((S,))
            log_p = build_log_posterior(estimator, thetas, task.dim_parameters, task.dim_data, prior)
            post[i-1] = log_p(theta_true, x_star)
        else:
            log_p = lambda theta: estimator(theta, x_star) + prior.log_prob(theta)
            post[i-1] = log_p(theta_true)
    return post

def sample(params, args, model_class, observation, task, reference_samples):
    # try:
    hamiltorch.set_random_seed(args.seed)
    prior = task.get_prior()
    simulator = task.get_simulator()
    x_star = observation.flatten()

    ### Set up model ###
    D_in = task.dim_parameters
    D_out = task.dim_data
    estimator = model_class(D_in, D_out, hidden_features=[params['hidden_size']] * 5, activation=nn.ELU)
    
    ### Sample ###
    if args.model_type == 'nre' or args.model_type == 'bnre':
        model_class = NRE
        ratio = False
    elif args.model_type == 'dnre':
        model_class = DNRE
        ratio = True

    model_path = 'model_epochs_' + str(params['epochs']) + 'hs_' + str(params['hidden_size']) + 'lr_' + str(params['lr']) +'.pt'
    model_path = os.path.join(args.path_dir, model_path)
    print('Loading pre-trained model... ',  model_path)
    estimator.load_state_dict(torch.load(model_path))
    estimator.eval()

    ### Sample ###
    prior = task.get_prior_dist()
    if args.metric == 'c2st':
        if args.hmc:
            if 'low' in list(task.get_prior_params().keys()):
                LOWER = task.get_prior_params()['low']
                UPPER = task.get_prior_params()['high']
                bounds = True
            else:
                LOWER = torch.zeros(reference_samples.shape[1]) #* 5
                UPPER = torch.ones(reference_samples.shape[1]) #* 5
                bounds = False
            posterior_samples, acceptance = hmc_nre(estimator, x_star, prior, n_chains = args.n_chains, step_size = params['step_size'], L = params['L'], n_steps = 2048, bounds = bounds, UPPER = UPPER, LOWER = LOWER, ratio = ratio, burn = 1024, device = 'cpu', adapt_step_size = args.adapt_step_size, desired_accept_rate = params['desired_acc_rate'])
            
            desired_acc_rate = params['desired_acc_rate'] + (1 - params['desired_acc_rate'])/2.
            while acceptance < 0.45 and desired_acc_rate < 0.99:
                print(f'Acceptance rate too low, one retry with {desired_acc_rate}')
                posterior_samples, acceptance = hmc_nre(estimator, x_star, prior, n_chains = args.n_chains, step_size = params['step_size'], L = params['L'], n_steps = 2048, bounds = bounds, UPPER = UPPER, LOWER = LOWER, ratio = ratio, burn = 1024, device = 'cpu', adapt_step_size = args.adapt_step_size, desired_accept_rate = desired_acc_rate)
                desired_acc_rate = desired_acc_rate + (1 - desired_acc_rate)/2.
            
            posterior_samples = posterior_samples[::4].reshape(-1, D_in)
        else:
            posterior_samples = mh_nre(estimator, x_star, sigma = params['sigma'], n_chains = args.n_chains, burn = 1024, n_samples_per_chain = 2048, thinning = 4, prior = prior, ratio = ratio)

        posterior_samples = posterior_samples[torch.randperm(len(posterior_samples))][:reference_samples.shape[0]]

        ### Evaluate ###
        from sbibm.metrics import c2st
        c2st_accuracy_nre = c2st(reference_samples, posterior_samples)
        print(c2st_accuracy_nre.item())

        return c2st_accuracy_nre.item(), posterior_samples
    
    elif args.metric == 'coverage':
        if 'low' in list(task.get_prior_params().keys()):
            LOWER = task.get_prior_params()['low']
            UPPER = task.get_prior_params()['high']
        else:
            LOWER = torch.zeros(reference_samples.shape[1]) #* 5
            UPPER = torch.ones(reference_samples.shape[1]) #* 5
            
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(args.device)
        
        estimator.to(device)
        if args.model_type == 'nre' or args.model_type == 'bnre':
            log_p = lambda theta, x: estimator(theta, x) + prior.log_prob(theta.cpu()).to(device)  # log p(theta | x) = log r(theta, x) + log p(theta)
        elif args.model_type == 'dnre':
            thetas = prior.sample((args.direct_M,))
            log_p = build_log_posterior(estimator, thetas, D_in, D_out, prior, device = device)
            
        loader = JointLoader(task.get_prior_dist(), simulator, batch_size=512, vectorized=True)
        H5Dataset.store(loader, 'test'+args.model_type+task.name+'.h5', size=args.coverage_number, overwrite=True)
        testset = H5Dataset('test'+args.model_type+task.name+'.h5')
        levels, coverages = expected_coverage_ni(log_p, testset, (LOWER, UPPER), device = device)
        return levels, coverages
    elif args.metric == 'posterior':
        
        return log_post_obs(estimator, prior, task, ratio, S = args.direct_M)
        
def main(args):
    if args.hmc:
        if args.adapt_step_size:
            path_dir = os.path.join(args.path_dir, 'hmc_adapt')
        else:
            path_dir = os.path.join(args.path_dir, 'hmc')
    else:
        path_dir = args.path_dir
    
    hamiltorch.set_random_seed(args.seed)
    task = sbibm.get_task(args.task)  # See sbibm.get_available_tasks() for all tasks
    prior = task.get_prior()
    simulator = task.get_simulator()
    
    if args.metric == 'c2st':
        result_file = os.path.join(path_dir, f"{args.model_type}_best_evaluation.joblib")
    elif args.metric == 'coverage':
        result_file = os.path.join(path_dir, f"{args.model_type}_best_evaluation_{args.metric}.joblib")
    elif args.metric == 'posterior':
        result_file = os.path.join(path_dir, f"{args.model_type}_best_evaluation_{args.metric}_s_{args.direct_M}.joblib")
        
    
    # Create results directory
    if os.path.exists(result_file) and args.force:
        print('Deleting Current Evaluation Results...')
        try:
            shutil.rmtree(result_file)
            os.makedirs(result_file)
        except:
            os.remove(result_file)
        observation_results = []
        observation_samples = []
        observations = []
        
    elif os.path.exists(result_file):
        dic = load(result_file)
        
        if len(dic['observations']) == 10 and args.metric == 'c2st':
            print('Evaluation Results Exist so exiting...')
            sys.exit()
        elif len(dic['observations']) < 10 and args.metric == 'c2st':
            observation_results = dic['c2st_accuracy']
            observation_samples = dic['observation_samples']
            observations = dic['observations']
        else:
            print('Evaluation Results Exist so exiting...')
            sys.exit()
    
    else:
        observation_results = []
        observation_samples = []
        observations = []
    
    best_params_path = os.path.join(path_dir, f"{args.model_type}_grid_search_result.joblib")
    best_params = load(best_params_path)
    
    # Select model
    if args.model_type == 'nre'  or args.model_type == 'bnre':
        estimator_class = NRE
        
    elif args.model_type == 'dnre':
        estimator_class = DNRE
    
    if args.metric == 'c2st':
        for i in range(10):
            print(f'Observation {i+1} out of 10')
            print(best_params)
            if i+1 in observations:
                print(f'Observation {i+1} already completed, moving on')
            else:
                reference_samples = task.get_reference_posterior_samples(num_observation=i+1)
                observation = task.get_observation(num_observation=i+1) 

                c2st_accuracy, posterior_samples = sample(best_params, args, estimator_class, observation, task, reference_samples)


                observation_results.append(c2st_accuracy)
                observation_samples.append(posterior_samples)
                observations.append(i+1)

                dic = {'c2st_accuracy': observation_results, 'observation_samples': observation_samples,
                       'observations': observations}

                # Save the results
                dump(dic, result_file)

    elif args.metric == 'coverage':
        reference_samples = task.get_reference_posterior_samples(num_observation=1)
        observation = task.get_observation(num_observation=1) 

        levels, coverages = sample(best_params, args, estimator_class, observation, task, reference_samples)

        dic = {'levels': levels, 'coverages': coverages,
               'observations': observations, 'testSamples': args.coverage_number, 'direct_samples': args.direct_M}

        # Save the results
        result_file = os.path.join(path_dir, f"{args.model_type}_best_evaluation_{args.metric}.joblib")
        dump(dic, result_file)
    
    elif args.metric == 'posterior':
        reference_samples = task.get_reference_posterior_samples(num_observation=1)
        observation = task.get_observation(num_observation=1) 

        log_posteriors = sample(best_params, args, estimator_class, observation, task, reference_samples)

        dic = {'log_posteriors': log_posteriors, 'direct_samples': args.direct_M}

        # Save the results
        result_file = os.path.join(path_dir, f"{args.model_type}_best_evaluation_{args.metric}_s_{args.direct_M}.joblib")
        dump(dic, result_file)
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate for all reference observations for nre models")
    parser.add_argument("--model_type", choices=["nre", "bnre", "dnre"], required=True, help="Model type to use for classification")
    parser.add_argument("--metric", choices=["c2st", "coverage", "posterior"], required=True, help="Metric to save as.")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--task', type=str, default='bernoulli_glm', help='Task to perform')
    parser.add_argument('--n_chains', type=int, default=1000, help='Number of chains')
    parser.add_argument('--direct_M', type=int, default=1000, help='Number of samples for direct approach')
    parser.add_argument('--coverage_number', type=int, default=512, help='Number of samples for direct approach')
    parser.add_argument('--path_dir', type=str, default='./test', help='Directory to benchmark results.')
    parser.add_argument('--hmc', action='store_true', default=False, help='Sample using hmc.')
    parser.add_argument('--force', action='store_true', default=False, help='Overwrite all results.')
    parser.add_argument('--adapt_step_size', action='store_true', default=False, help='Whether to adapt acceptance rate.')
    parser.add_argument('--desired_acc_rate', type=float, default=0.75, help='Desired acceptance rate.')
    parser.add_argument('--device', type=int, help='GPU device', default=0)
    
    args = parser.parse_args()
    main(args)

    
# python benchmark_evaluate.py --model_type nre --task two_moons --path_dir ./benchmark_results/two_moons/nre_10000 --device 0 --metric coverage
