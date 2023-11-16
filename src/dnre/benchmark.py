import os
import argparse

import sbibm
import hamiltorch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
import uuid

from lampe.data import JointLoader, JointDataset
from lampe.inference import NRE, NRELoss, BNRELoss
from dnre.inference import *
from lampe.plots import nice_rc, corner, mark_point
from lampe.utils import GDStep
from sklearn.model_selection import ParameterGrid

from joblib import dump, load
import concurrent.futures
import shutil


def train(estimator, loss_fun, dataset, dataset_val, lr, epochs, device, return_losses=False):
    estimator = estimator.train().to(device)
    loss = loss_fun(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=lr)
    step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping
    loss_list = []
    val_loss_list = []
    best_val_loss = float("inf")
    with tqdm(range(epochs), unit='epoch') as tq:
        for epoch in tq:
            losses = torch.stack([
                step(loss(theta.to(device), x.to(device)))
                for theta, x in dataset # 256 batches per epoch
            ])
            with torch.no_grad():
                losses_val = torch.stack([
                loss(theta.to(device), x.to(device))
                for theta, x in dataset_val # 256 batches per epoch
            ])
            loss_list.append(losses.sum().item() / len(dataset))
            val_loss_list.append(losses_val.sum().item() / len(dataset_val))
            
            if val_loss_list[-1] < best_val_loss:
                best_val_loss = val_loss_list[-1]
                best_estimator = deepcopy(estimator)
            
            tq.set_postfix(loss=loss_list[-1], val = val_loss_list[-1])
        
    print('Best val loss: ', best_val_loss)
    if return_losses:
        return best_estimator.cpu(), loss_list, val_loss_list
    else:
        return best_estimator.cpu()

def train_and_sample(params, args, model_class, dataset, dataset_val, observation, task, reference_samples):
    if args.hmc:
        if args.adapt_step_size:
            save_dir = os.path.join(args.save_dir, 'hmc_adapt')
        else:
            save_dir = os.path.join(args.save_dir, 'hmc')
    else:
        save_dir = args.save_dir
    
    # try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)
    hamiltorch.set_random_seed(args.seed)
    prior = task.get_prior()

    x_star = observation.flatten()

    ### Set up model ###
    D_in = dataset.theta.shape[1]
    D_out = dataset.x.shape[1]
    estimator = model_class(D_in, D_out, hidden_features=[params['hidden_size']] * 5, activation=nn.ELU)
    
    ### Train ###
    model_path = 'model_epochs_' + str(params['epochs']) + 'hs_' + str(params['hidden_size']) + 'lr_' + str(params['lr']) +'.pt'
    model_path = os.path.join(args.save_dir, model_path)
    if args.model_type == 'nre':
        loss_fun = NRELoss
        ratio = False
    elif args.model_type == 'bnre':
        loss_fun = BNRELoss
        ratio = False
    elif args.model_type == 'dnre':
        loss_fun = DNRELoss
        ratio = True
    
    if os.path.exists(model_path): 
        print('Loading pre-trained model... ',  model_path)
        estimator.load_state_dict(torch.load(model_path))
        estimator.eval()
    else:
        
        estimator = train(estimator, loss_fun, dataset, dataset_val, lr = params['lr'], epochs = params['epochs'], device = device)
        torch.save(estimator.state_dict(), model_path)
    
    ### Sample ###
    prior = task.get_prior_dist()
    
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
        posterior_samples = posterior_samples[::4].reshape(-1, D_in)
    else:
        posterior_samples = mh_nre(estimator, x_star, sigma = params['sigma'], n_chains = args.n_chains, burn = 1024, n_samples_per_chain = 2048, thinning = 4, prior = prior, ratio = ratio)
    
    posterior_samples = posterior_samples[torch.randperm(len(posterior_samples))][:reference_samples.shape[0]]

    ### Evaluate ###
    from sbibm.metrics import c2st
    c2st_accuracy_nre = c2st(reference_samples, posterior_samples)
    print(c2st_accuracy_nre.item())
    
    ### Save ###
    save_loc = os.path.join(save_dir, f"c2st_{round(c2st_accuracy_nre.item(), 4)}_grid_" + args.model_type + "_" + uuid.uuid4().hex + ".joblib")

    dump(params, save_loc)
    
    return params, c2st_accuracy_nre.item()

def main(args):
    hamiltorch.set_random_seed(args.seed)
    task = sbibm.get_task(args.task)  # See sbibm.get_available_tasks() for all tasks
    prior = task.get_prior()
    simulator = task.get_simulator()
    observation = task.get_observation(num_observation=args.num_observation)  # 10 per task

    thetas = prior(num_samples=args.n_training_samples)
    xs = simulator(thetas)

    reference_samples = task.get_reference_posterior_samples(num_observation=args.num_observation)
    reference_samples = reference_samples[:args.n_chains]
    
    # Training Data
    dataset = JointDataset(thetas, xs, batch_size=args.batch_size, shuffle=True)
    
    # Validation Data
    thetas_val = prior(num_samples=args.n_val_samples)
    xs_val = simulator(thetas_val)
    dataset_val = JointDataset(thetas_val, xs_val, batch_size=args.batch_size, shuffle=False)
    
    # Define the model search space and other hyperparameters
    
    param_grid = {
        'hidden_size': [64],
        'lr': [1e-4, 5e-4, 1e-3],
        'epochs': [20000],
        'sigma': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        }
    
    if args.hmc:
        param_grid_add = {'L': [1, 5, 10],
                          'step_size': [0.001, 0.005, 0.01, 0.03]
                          }
        if args.adapt_step_size:
            param_grid_add.pop('step_size')
            param_grid_add['step_size'] = [0.1]
            param_grid_add['desired_acc_rate'] = [0.5, 0.6, 0.7, 0.8]
            
        param_grid.pop("sigma")
        param_grid.update(param_grid_add)
    
    # Select model
    if args.model_type == 'nre'  or args.model_type == 'bnre':
        estimator_class = NRE
        
    elif args.model_type == 'dnre':
        estimator_class = DNRE
    
    if args.hmc:
        if args.adapt_step_size:
            save_dir = os.path.join(args.save_dir, 'hmc_adapt')
        else:
            save_dir = os.path.join(args.save_dir, 'hmc')
    else:
        save_dir = args.save_dir
    # Create results directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(save_dir) and args.force:
        print('Deleting Current Directory...')
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    result_file = os.path.join(save_dir, f"{args.model_type}_grid_search_result.joblib")
    
    # Load previous results if available
    if os.path.exists(result_file) and not args.force:
        print(f"Resuming previous {args.model_type} grid search...")
        best_params = load(result_file)
        best_c2st_accuracy = load(os.path.join(save_dir, f"{args.model_type}_grid_search_result_c2st_accuracy.joblib"))
        
        params_list = load(os.path.join(save_dir, f"{args.model_type}_params_list.joblib"))
        
    else:
        print(f"Starting new {args.model_type} grid search...")
        best_c2st_accuracy = 1.0
        best_params = None
        params_list = []

    for i, params in enumerate(ParameterGrid(param_grid)):
        print(f'Iteration {i} out of {len(ParameterGrid(param_grid))}')
        print(params)
        if params in params_list:
            print('Already completed... Moving to next one')
        else:
            params, c2st_accuracy = train_and_sample(params, args, estimator_class, dataset, dataset_val, observation, task, reference_samples)
            if (c2st_accuracy - 0.5) ** 2 < (best_c2st_accuracy - 0.5) ** 2:
                best_c2st_accuracy = c2st_accuracy
                best_params = params
            # Save the best hyperparameters
            dump(best_params, result_file)
            dump(best_c2st_accuracy, os.path.join(save_dir, f"{args.model_type}_grid_search_result_c2st_accuracy.joblib"))
            dump(args, os.path.join(save_dir, f"{args.model_type}_args.joblib"))
            params_list.append(params)
            dump(params_list, os.path.join(save_dir, f"{args.model_type}_params_list.joblib"))

    print(f"Best model: {best_params}")
    print(f"Test accuracy: {best_c2st_accuracy}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for nre models")
    parser.add_argument("--model_type", choices=["nre", "bnre", "dnre"], required=True, help="Model type to use for classification")
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--task', type=str, default='bernoulli_glm', help='Task to perform')
    parser.add_argument('--n_training_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--num_observation', type=int, default=1, help='sbib observation (x_star)')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    parser.add_argument('--n-val_samples', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--n_chains', type=int, default=1000, help='Number of chains')
    parser.add_argument('--max_workers', type=int, default=1, help='Maximum number of workers')
    parser.add_argument('--save_dir', type=str, default='./test', help='Directory to save all restuls.')
    parser.add_argument('--force', action='store_true', default=False, help='Overwrite all results.')
    parser.add_argument('--hmc', action='store_true', default=False, help='Sample using hmc.')
    parser.add_argument('--adapt_step_size', action='store_true', default=False, help='Whether to adapt acceptance rate.')
    parser.add_argument('--desired_acc_rate', type=float, default=0.75, help='Desired acceptance rate.')
    parser.add_argument('--device', type=int, help='GPU device', default=0)
    
    args = parser.parse_args()
    main(args)

    
# python benchmark.py --model_type nre --task two_moons --save_dir ./benchmark_results/two_moons/nre_10000 --device 0