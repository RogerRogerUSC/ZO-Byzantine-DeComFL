import torch
import numpy as np
from typing import List, Tuple

def Bayesian_estimate(g: List[List[torch.Tensor]], max_iter: int = 10,tol: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor, float, List[List[torch.Tensor]], List[List[torch.Tensor]]]:
    """
    Perform Bayesian estimation to recover the true mean, variance, and identify modified gradient scalars.
    
    Parameters:
        g (List[List[torch.Tensor]]): A 2D list of gradient scalars.
            - Outer list: clients (M clients)
            - Inner list: gradient scalars from that client (P*K scalars)
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.
    
    Returns:
        mu (torch.Tensor): Estimated true mean.
        sigma2 (torch.Tensor): Estimated true variance.
        pi (float): Estimated proportion of modified gradient scalars.
        flags_list (List[List[torch.Tensor]]): 2D list of flags (0: unmodified, 1: modified) with same structure as g.
        recovered_list (List[List[torch.Tensor]]): 2D list of recovered gradient scalars (modified ones replaced).
    """
    M = len(g)
    if M == 0:
        raise ValueError("Input list g is empty.")
    P = len(g[0])
    flattened_list = []
    for client in g:
        if len(client) != P:
            raise ValueError("All inner lists must have the same length.")
        flattened_list.extend(client)
    g_flat = torch.stack(flattened_list)

    # Initialize parameters on the flattened tensor
    flags = torch.zeros_like(g_flat)
    mu = torch.mean(g_flat)
    sigma2 = torch.var(g_flat, unbiased=True)
    pi = 0.5

    for _ in range(max_iter):
        # Sample new pi based on current flags
        pi_new = sample_pi(flags, alpha_prior=1.0, beta_prior=1.0)
        
        # Sample new flags for each scalar using the current mu, sigma2, and pi_new
        flags_new = sample_flag(g_flat, mu, sigma2, pi_new)
        
        # Recover modified scalars by resampling them from N(mu, sigma2)
        g_recovered = g_flat.clone()
        for idx in range(g_flat.numel()):
            if flags_new[idx].item() == 1:
                g_recovered[idx] = sample_scalar(g_flat[idx], 1, mu, sigma2)
        
        # Update mu and sigma2 using only unmodified (or recovered) scalars
        mu_new = sample_mean(g_recovered, flags_new)
        sigma2_new = sample_variance(g_recovered, flags_new, mu_new)
        
        # Check for convergence (if changes in parameters are below tolerance, break)
        if (torch.abs(mu_new - mu) < tol and 
            torch.abs(sigma2_new - sigma2) < tol and 
            abs(pi_new - pi) < tol):
            mu, sigma2, pi, flags = mu_new, sigma2_new, pi_new, flags_new
            g_flat = g_recovered
            break
        
        # Otherwise, update parameters and continue iterating
        mu, sigma2, pi, flags = mu_new, sigma2_new, pi_new, flags_new
        g_flat = g_recovered

    # Unflatten the 1D recovered tensor and flags back to the original 2D structure
    recovered_list = []
    flags_list = []
    for i in range(M):
        # Get the slice corresponding to client i
        client_slice = g_flat[i * P : (i + 1) * P]
        client_flags = flags[i * P : (i + 1) * P]
        # Convert each element to a scalar tensor and collect into a list
        recovered_list.append([client_slice[j] for j in range(P)])
        flags_list.append([client_flags[j] for j in range(P)])
    
    return mu, sigma2, pi, flags_list, recovered_list

def sample_pi(flags: torch.Tensor, alpha_prior: float = 1.0, beta_prior: float = 1.0) -> float:
    """
    Sample the proportion (pi) of modified gradient scalars using a Beta posterior.
    """
    n = flags.numel()
    sum_z = torch.sum(flags).item()
    alpha_post = alpha_prior + sum_z
    beta_post = beta_prior + n - sum_z
    beta_dist = torch.distributions.Beta(alpha_post, beta_post)
    return beta_dist.sample().item()

def sample_flag(g: torch.Tensor, mu: torch.Tensor, sigma2: torch.Tensor, pi: float) -> torch.Tensor:
    """
    Sample modification flags for each gradient scalar.
    """
    norm_pdf = (1.0 / torch.sqrt(2 * torch.pi * sigma2)) * torch.exp(-((g - mu) ** 2) / (2 * sigma2))
    p_modified = pi / (pi + (1 - pi) * norm_pdf)
    flags = torch.bernoulli(p_modified)
    return flags

def sample_scalar(g: torch.Tensor, flag: int, mu: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """
    Recover a gradient scalar. If unmodified (flag == 0), return the original value.
    If modified (flag == 1), resample from N(mu, sigma2).
    """
    if flag == 0:
        return g
    else:
        return torch.normal(mu, torch.sqrt(sigma2))

def sample_mean(g: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
    """
    Sample a new estimate for the true mean using only unmodified gradient scalars.
    """
    unmodified = g[flags == 0]
    if unmodified.numel() > 0:
        unmodified_np = unmodified.cpu().numpy()
        mu_n_np = np.mean(unmodified_np)
        if unmodified_np.size > 1:
            var_np = np.var(unmodified_np, ddof=1)
        else:
            var_np = 0.0
        sigma_n2 = var_np / unmodified_np.size
        mu_n = torch.tensor(mu_n_np, dtype=g.dtype, device=g.device)
        std = torch.sqrt(torch.tensor(sigma_n2, dtype=g.dtype, device=g.device))
        if std < 1e-6:
            return mu_n
        return torch.normal(mu_n, std)
    else:
        return torch.mean(g)

def sample_variance(g: torch.Tensor, flags: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """
    Sample a new estimate for the true variance using only unmodified gradient scalars.
    
    """
    unmodified = g[flags == 0]
    n_x = unmodified.numel()
    if n_x == 0:
        return torch.var(g, unbiased=True)
    
    alpha_prior = 1.0
    beta_prior = 1.0
    alpha_post = alpha_prior + n_x / 2.0
    beta_post = beta_prior + 0.5 * torch.sum((unmodified - mu) ** 2)
    
    inv_gamma_dist = torch.distributions.InverseGamma(alpha_post, beta_post)
    return inv_gamma_dist.sample()

if __name__ == "__main__":
    observed_g = [[torch.tensor(val) for val in [10.0]] for _ in range(10)]
    observed_g[3][0] = torch.tensor(15.0)
    observed_g[7][0] = torch.tensor(14.5)
    
    mu_est, sigma2_est, pi_est, flags_est, recovered_g = Bayesian_estimate(observed_g, max_iter=20, tol=1e-3)
    
    print("Estimated Mean (mu):", mu_est.item())
    print("Estimated Variance (sigma^2):", sigma2_est.item())
    print("Estimated Proportion Modified (pi):", pi_est)
    print("Final Flags (per client):", flags_est)
    print("Recovered Gradient Scalars (per client):", recovered_g)
