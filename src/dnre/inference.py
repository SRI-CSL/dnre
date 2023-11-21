r"""Inference components such as estimators, training losses and MCMC samplers.

Code derived from Lampe: https://github.com/francois-rozet/lampe/tree/master/lampe
and hamiltorch: https://github.com/AdamCobb/hamiltorch

"""

__all__ = [
    'MetropolisHastings',
    'DNRE',
    'DNRELoss',
    'train_estimator',
    'hmc_nre',
    'mh_nre',
    'MultivariateUniform'
]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional as F_autograd
import torch.optim as optim
from torch.distributions import Distribution

from itertools import islice
from torch import Tensor, BoolTensor, Size
from typing import *

from zuko.distributions import Distribution, DiagNormal, NormalizingFlow
from zuko.transforms import FreeFormJacobianTransform
from zuko.utils import broadcast

from lampe.nn import MLP
from lampe.utils import GDStep
from pyknos.nflows.nn import nets
import hamiltorch

from copy import deepcopy

from tqdm import tqdm


class DNRE(nn.Module):

    def __init__(
        self,
        theta_dim: int,
        x_dim: int,
        build: Callable[[int, int], nn.Module] = MLP,
        model: str = 'mlp', # 'resnet'
        **kwargs,
    ):
        super().__init__()
        
        if model == 'mlp':
            self.net = build(theta_dim * 2 + x_dim, 1, **kwargs)
        elif model =='resnet':
            self.net = nets.ResidualNet(
                in_features=theta_dim * 2 + x_dim,
                out_features=1,
                hidden_features=50,
                context_features=None,
                num_blocks=2,
                activation=torch.relu,
                dropout_probability=0.0,
                use_batch_norm=False,
            )

    def forward(self, theta: Tensor, theta_prime: Tensor, x: Tensor) -> Tensor:
        r"""
        Arguments:
            theta: The parameters :math:`\theta`, with shape :math:`(*, D)`.
            x: The observation :math:`x`, with shape :math:`(*, L)`.

        Returns:
            The log-ratio :math:`\log r_\phi(\theta, x)`, with shape :math:`(*,)`.
        """
        # print(f'0 {theta.shape}, {x.shape}')
        # theta, x = broadcast(theta, x, ignore=1)
        # print(f'{theta.shape}, {x.shape}')

        return self.net(torch.cat((theta, theta_prime, x), dim=-1)).squeeze(-1)

class DNRELoss(nn.Module):

    def __init__(self, estimator: nn.Module):
        super().__init__()

        self.estimator = estimator

    def forward(self, theta: Tensor, x: Tensor) -> Tensor:
        
        shift = torch.randint(1,theta.shape[0],(1,)).item()
        
        theta_prime = torch.roll(theta, shift, dims=0)

        log_r = self.estimator(theta, theta_prime, x)
        # reverse ordering:
        log_r_prime = self.estimator(theta_prime, theta, x)

        l1 = -F.logsigmoid(log_r).mean()
        l0 = -F.logsigmoid(-log_r_prime).mean()

        return (l1 + l0) / 2


class MetropolisHastings(object):
    r"""Creates a batched Metropolis-Hastings sampler.

    Metropolis-Hastings is a Markov chain Monte Carlo (MCMC) sampling algorithm used to
    sample from intractable distributions :math:`p(x)` whose density is proportional to
    a tractable function :math:`f(x)`, with :math:`x \in \mathcal{X}`. The algorithm
    consists in repeating the following routine for :math:`t = 1` to :math:`T`, where
    :math:`x_0` is the initial sample and :math:`q(x' | x)` is a pre-defined transition
    distribution.

    .. math::
        1. ~ & x' \sim q(x' | x_{t-1}) \\
        2. ~ & \alpha \gets \frac{f(x')}{f(x_{t-1})}
            \frac{q(x_{t-1} | x')}{q(x' | x_{t-1})} \\
        3. ~ & u \sim \mathcal{U}(0, 1) \\
        4. ~ & x_t \gets \begin{cases}
            x' & \text{if } u \leq \alpha \\
            x_{t-1} & \text{otherwise}
        \end{cases}

    Asymptotically, i.e. when :math:`T \to \infty`, the distribution of samples
    :math:`x_t` is guaranteed to converge towards :math:`p(x)`. In this implementation,
    a Gaussian transition :math:`q(x' | x) = \mathcal{N}(x'; x, \Sigma)` is used, which
    can be modified by sub-classing :class:`MetropolisHastings`.

    Wikipedia:
        https://wikipedia.org/wiki/Metropolis-Hastings_algorithm

    Arguments:
        x_0: A batch of initial points :math:`x_0`, with shape :math:`(*, L)`.
        f: A function :math:`f(x)` proportional to a density function :math:`p(x)`.
        log_f: The logarithm :math:`\log f(x)` of a function proportional
            to :math:`p(x)`.
        sigma: The standard deviation of the Gaussian transition.
            Either a scalar or a vector.

    Example:
        >>> x_0 = torch.randn(128, 7)
        >>> log_f = lambda x: -(x**2).sum(dim=-1) / 2
        >>> sampler = MetropolisHastings(x_0, log_f=log_f, sigma=0.5)
        >>> samples = [x for x in sampler(256, burn=128, step=4)]
        >>> samples = torch.stack(samples)
        >>> samples.shape
        torch.Size([32, 128, 7])
    """

    def __init__(
        self,
        x_0: Tensor,
        f: Callable[[Tensor], Tensor] = None,
        log_f: Callable[[Tensor], Tensor] = None,
        log_y_x: Callable[[Tensor], Tensor] = None,
        sigma: Union[float, Tensor] = 1.0,
    ):
        super().__init__()

        self.x_0 = x_0

        assert f is not None or log_f is not None or log_y_x is not None, \
            "Either 'f' or 'log_f' has to be provided."

        if f is None and log_f is not None:
            self.log_f = log_f
            self.log_y_x = None
        elif f is not None:
            self.log_f = lambda x: f(x).log()
            self.log_y_x = None
        else:
            self.log_y_x = log_y_x

        self.sigma = sigma

    def q(self, x: Tensor) -> Distribution:
        return DiagNormal(x, torch.ones_like(x) * self.sigma)

    @property
    def symmetric(self) -> bool:
        return True

    def __iter__(self) -> Iterator[Tensor]:
        x = self.x_0

        # log f(x)
        if self.log_y_x is None:
            log_f_x = self.log_f(x)

        while True:
            # y ~ q(y | x)
            y = self.q(x).sample()

            # log f(y)
            if self.log_y_x is None:
                log_f_y = self.log_f(y)

            #     f(y)   q(x | y)
            # a = ---- * --------
            #     f(x)   q(y | x)
            
            if self.log_y_x is None:
                log_a = log_f_y - log_f_x
            else:
                log_a = self.log_y_x(y,x)

            if not self.symmetric:
                log_a = log_a + self.q(y).log_prob(x) - self.q(x).log_prob(y)

            a = log_a.exp()

            # u in [0; 1]
            u = torch.rand(a.shape).to(a)

            # if u < a, x <- y
            # else x <- x
            mask = u < a

            x = torch.where(mask.unsqueeze(-1), y, x)
            if self.log_y_x is None:
                log_f_x = torch.where(mask, log_f_y, log_f_x)

            yield x

    def __call__(self, stop: int, burn: int = 0, step: int = 1) -> Iterable[Tensor]:
        return islice(self, burn, stop, step)

    
def hamiltonian(params, momentum, log_prob_func, inv_mass=None):
    """Computes the Hamiltonian as a function of the parameters and the momentum.

    Parameters
    ----------
    params : torch.tensor
        Flat vector of model parameters: shape (D,), where D is the dimensionality of the parameters.
    momentum : torch.tensor
        Flat vector of momentum, corresponding to the parameters: shape (D,), where D is the dimensionality of the parameters.
    log_prob_func : function
        A log_prob_func must take a 1-d vector of length equal to the number of parameters that are being sampled.
    inv_mass : torch.tensor or list
        The inverse of the mass matrix. The inv_mass matrix is related to the covariance of the parameter space (the scale we expect it to vary). Currently this can be set
        to either a diagonal matrix, via a torch tensor of shape (D,), or a full square matrix of shape (D,D). There is also the capability for some
        integration schemes to implement the inv_mass matrix as a list of blocks. Hope to make that more efficient.

    Returns
    -------
    torch.tensor
        Returns the value of the Hamiltonian: shape (1,).

    """

    log_prob = log_prob_func(params)

    if hamiltorch.util.has_nan_or_inf(log_prob):
        print('Invalid log_prob: {}, params: {}'.format(log_prob, params))
        raise hamiltorch.util.LogProbError()


    potential = -log_prob#/normalizing_const
    # Multiple chains such that momentum is C x D
    if potential.numel() > 1:
        if inv_mass is None:
            kinetic = 0.5 * (momentum * momentum).sum(1) # sum over dimension
        else:
            raise NotImplementedError()
    else:
        if inv_mass is None:
            kinetic = 0.5 * torch.dot(momentum, momentum)#/normalizing_const
        else:
            if type(inv_mass) is list:
                i = 0
                kinetic = 0
                for block in inv_mass:
                    it = block[0].shape[0]
                    kinetic = kinetic +  0.5 * torch.matmul(momentum[i:it+i].view(1,-1),torch.matmul(block,momentum[i:it+i].view(-1,1))).view(-1)#/normalizing_const
                    i += it
            #Assum G is diag here so 1/Mass = G inverse
            elif len(inv_mass.shape) == 2:
                kinetic = 0.5 * torch.matmul(momentum.view(1,-1),torch.matmul(inv_mass,momentum.view(-1,1))).view(-1)#/normalizing_const
            else:
                kinetic = 0.5 * torch.dot(momentum, inv_mass * momentum)#/normalizing_const
    hamiltonian = potential + kinetic

    return hamiltonian
    
    
def hmc_nre(estimator, x_star, prior, n_chains, step_size, L, n_steps, bounds = False, UPPER = None, LOWER = None, ratio = False, burn = 1024, pass_grad = None, device = 'cpu', adapt_step_size = False, desired_accept_rate = 0.8, min_step_size = 1e-8, init_theta = None):   
    step_size_init = step_size
    H_t = 0.
    eps_bar = 1.
    rho = -torch.ones(10)*1e4 # initialize as a rejection
    
    j = 0
    estimator.to(device).eval()
    
    if ratio:
        x_star_repeat = x_star.view(1,-1).repeat(n_chains,1).to(device)
        log_p = lambda theta, theta_prime: estimator(theta, theta_prime, x_star_repeat) + prior.log_prob(theta) - prior.log_prob(theta_prime)
    else:
        x_star = x_star.to(device)
        log_p = lambda theta: estimator(theta, x_star) + prior.log_prob(theta)  # p(theta | x) = r(theta, x) p(theta)
    if init_theta is None:
        theta = prior.sample((n_chains,)).to(device)
    else:
        assert init_theta.shape[0] == n_chains
        theta = init_theta.clone()
    dim = theta.shape[1]
    samples = torch.zeros(n_steps, n_chains, dim)
    acceptance = 0
    for i in tqdm(range(n_steps + burn + 1)):
        p = torch.randn(n_chains, dim).to(device)
        try:
            theta, p, acc, rho = HMC_step(log_p, theta, p, n_chains, step_size, L = L, bounds = bounds, UPPER = UPPER, LOWER = LOWER, ratio = ratio, pass_grad = pass_grad, prior = prior, device=device)
        except hamiltorch.util.LogProbError:
            acc = torch.zeros(n_chains)
        if i > burn:
            samples[j] = theta.clone().cpu()
            j+=1
        elif adapt_step_size:
            if i < burn:
                # import pdb; pdb.set_trace()
                rho_log_sum_exp_mean = torch.logsumexp(rho.detach(), 0) - np.log(n_chains)
                step_size, eps_bar, H_t = hamiltorch.samplers.adaptation(rho_log_sum_exp_mean, i, step_size_init, H_t, eps_bar, desired_accept_rate=desired_accept_rate)
            if i == burn:
                step_size = max(eps_bar, min_step_size)
                print('Final Adapted Step Size: ',step_size)
            
        acceptance += acc.sum().item()
    print(acceptance/ ((n_steps + burn) * n_chains))
    return samples, acceptance/ ((n_steps + burn) * n_chains)

def HMC_step(log_prob_func, theta, p, n_chains, step_size, L, bounds = False, UPPER = None, LOWER = None, ratio = False, pass_grad = None, prior = None, device = 'cpu'):
    
    if ratio:
        theta_prime = prior.sample((n_chains,)).to(device)
        log_prob_func_sum = lambda x: log_prob_func(x, theta_prime).sum()
    else:
        log_prob_func_sum = lambda x: log_prob_func(x).sum() # For dealing with chains
    if ratio:
        theta_prev = theta.clone()
        p_prev = p.clone()
    else:
        H0 = hamiltonian(theta, p, log_prob_func)
    ret_params, ret_momenta = hamiltorch.samplers.leapfrog(params = theta, momentum = p, log_prob_func = log_prob_func_sum, steps=L, step_size=step_size, pass_grad = pass_grad)
    theta_proposed = ret_params[-1].detach().clone()
    p_proposed = -ret_momenta[-1].detach().clone() # Negate
    
    if bounds:
        # Check prior bounds on theta_proposed and build mask per chain
        bound_mask = torch.logical_or(theta_proposed >= UPPER, theta_proposed <= LOWER)
        bound_mask = torch.any(bound_mask, 1)
        theta_proposed[bound_mask] = (UPPER + LOWER)/2. # Set to a dummy variable for MH Step but reject using mask after
    
    if ratio:
        H_diff = direct_nre_mhhmc(theta_proposed, theta_prev, p_proposed, p_prev, log_prob_func)
        rho = torch.min(torch.stack([H_diff.cpu(), torch.zeros(n_chains)]), 0)[0]
    else:
        H1 = hamiltonian(theta_proposed, p_proposed, log_prob_func)
        rho = torch.min(torch.stack([-H1.cpu() + H0.cpu(), torch.zeros(n_chains)]), 0)[0]
    accept_mask = (rho >= torch.log(torch.rand(n_chains))).to(device)
    
    if bounds:
        # Incorporating bounds:
        accept_mask = torch.logical_and(accept_mask, torch.logical_not(bound_mask)).to(device)
    
    theta[accept_mask] = theta_proposed[accept_mask].clone()
    p[accept_mask] = p_proposed[accept_mask].clone()
    return theta, p, sum(accept_mask), rho

def direct_nre_mhhmc(theta_proposed, theta_prev, p_proposed, p_prev, log_prob_func):
    
    # -U(q') + U(q)
    potential_diff = log_prob_func(theta_proposed, theta_prev)
    
    kinetic_prev = 0.5 * (p_prev * p_prev).sum(1)
    kinetic_prop = 0.5 * (p_proposed * p_proposed).sum(1)
    
    H_diff = potential_diff - kinetic_prop + kinetic_prev
    
    return H_diff
    
def train_estimator(estimator, loss_fun, epochs, dataset, dataset_val, lr, device='cpu'):
    loss = loss_fun(estimator)
    optimizer = optim.AdamW(estimator.parameters(), lr=lr)
    step = GDStep(optimizer, clip=1.0)  # gradient descent step with gradient clipping
    estimator.train()
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
            
    return best_estimator, loss_list, val_loss_list

def mh_nre(estimator, x_star, sigma = 0.5, n_chains = 1024, burn = 1024, n_samples_per_chain = 2048, thinning = 4, prior = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2)), ratio = False):
    estimator.cpu().eval()

    with torch.no_grad():
        theta_0 = prior.sample((n_chains,))  # 1024 concurrent Markov chains
        if ratio:
            x_star = x_star.view(1,-1).repeat(n_chains,1)
            if isinstance(estimator, DNRE):
                log_theta_theta_p = lambda theta, theta_prime: estimator(theta, theta_prime, x_star) + prior.log_prob(theta) - prior.log_prob(theta_prime)
            elif isinstance(estimator, DNRE_sigma):
                log_theta_theta_p = lambda theta, theta_prime: estimator(theta, theta_prime, x_star, torch.ones(n_chains,1)*sigma) + prior.log_prob(theta) - prior.log_prob(theta_prime)
            sampler = MetropolisHastings(theta_0, log_y_x=log_theta_theta_p, sigma=sigma)
            samples = torch.cat([
                theta
                for theta in sampler(n_samples_per_chain, burn=burn, step=thinning)
            ])
        else:
            log_p = lambda theta: estimator(theta, x_star) + prior.log_prob(theta)  # p(theta | x) = r(theta, x) p(theta)

            sampler = MetropolisHastings(theta_0, log_f=log_p, sigma=sigma)
            samples = torch.cat([
                theta
                for theta in sampler(n_samples_per_chain, burn=burn, step=thinning)
            ])
    return samples

class MultivariateUniform(Distribution):
    def __init__(self, low, high):
        self.low = torch.tensor(low)
        self.high = torch.tensor(high)
        self.dim = len(low)
        super(MultivariateUniform, self).__init__(batch_shape=(self.dim,), event_shape=(self.dim,))

    def sample(self, sample_shape=torch.Size()):
        return torch.rand(sample_shape + torch.Size([self.dim])) * (self.high - self.low) + self.low

    def log_prob(self, value):
        inside_interval = torch.logical_and(value >= self.low, value <= self.high)
        inside_interval = inside_interval.all(dim=-1)
        interval_volume = torch.prod(self.high - self.low)
        return torch.where(inside_interval, -torch.log(interval_volume), torch.tensor(-float('inf')))
