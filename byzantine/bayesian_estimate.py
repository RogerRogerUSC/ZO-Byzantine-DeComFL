from __future__ import annotations

import math
from typing import List, Tuple

import torch


def Bayesian_estimate(
    g: List[List[torch.Tensor]],
    *,
    max_iter: int = 10,
    tol: float = 1e-3,
) -> Tuple[
    torch.Tensor,                     # global posterior mean   (0-D tensor)
    torch.Tensor,                     # global posterior var    (0-D tensor)
    float,                            # global posterior pi     (float)
    List[List[torch.Tensor]],         # nested flags            (0-D tensors)
    List[torch.Tensor],               # [tensor(num_pert)]
]:
    # Unpack legacy payloads: a single 1-D tensor per client → list of 0-D tensors (one per perturbation).
    if g and all(len(client) == 1 and client[0].ndim == 1 for client in g):
        P = g[0][0].numel()
        unpacked: List[List[torch.Tensor]] = []
        for idx, client_vec in enumerate(g):
            vec = client_vec[0]
            if vec.numel() != P:
                raise ValueError(
                    f"Client {idx} sent vector length {vec.numel()} (expected {P})."
                )
            unpacked.append([vec[j].clone() for j in range(P)])  # 0-D each
        g = unpacked

    # Shape checks
    if not g:
        raise ValueError("Input list g is empty.")

    M, P = len(g), len(g[0])
    for i, client in enumerate(g):
        if len(client) != P:
            raise ValueError(f"Client {i} has {len(client)} directions; expected {P}.")
        for j, t in enumerate(client):
            if t.numel() != 1 or t.ndim != 0:
                raise ValueError(
                    f"Tensor at g[{i}][{j}] must be 0-D; got shape {tuple(t.shape)}."
                )

    # Stack into (M, P) matrix for vectorised maths
    device, dtype = g[0][0].device, g[0][0].dtype
    g_mat = torch.stack([torch.stack(client) for client in g]) # (M, P)
    g_vec = g_mat.flatten()  # (M·P,)

    flags = torch.zeros_like(g_vec) # corruption flags (0 = honest)
    mu     = g_vec.mean()
    sigma2 = (
        g_vec.var(unbiased=True) if g_vec.numel() > 1
        else torch.tensor(0.0, device=device, dtype=dtype)
    )
    pi = 0.5 # initialise to Beta(1,1) mean

    for _ in range(max_iter):
        # sample π
        pi_new = _sample_pi(flags)

        # sample flags z_i
        flags_new = _sample_flag(g_vec, mu, sigma2, pi_new)

        # recover corrupted values
        rec = g_vec.clone()
        mask = flags_new.bool()
        if mask.any():
            rec[mask] = torch.normal(
                mu,
                torch.sqrt(sigma2).clamp_min(1e-12),
                size=(int(mask.sum().item()),),
                device=device,
                dtype=dtype,
            )

        # sample μ and σ² from conjugate posteriors
        mu_new     = _sample_mean(rec, flags_new)
        sigma2_new = _sample_variance(rec, flags_new, mu_new)

        # Convergence check
        if (
            torch.abs(mu_new - mu) < tol
            and torch.abs(sigma2_new - sigma2) < tol
            and abs(pi_new - pi) < tol
        ):
            mu, sigma2, pi, flags = mu_new, sigma2_new, pi_new, flags_new
            g_vec = rec
            break

        mu, sigma2, pi, flags = mu_new, sigma2_new, pi_new, flags_new
        g_vec = rec

    # Reshape outputs to match caller expectations
    flags_mat = flags.reshape(M, P)
    flags_nested = [[flags_mat[i, j].unsqueeze(0) for j in range(P)] for i in range(M)]

    # Per-direction means of *recovered* scalars (needed downstream)
    rec_mat = g_vec.reshape(M, P)
    global_grad_scalar = [rec_mat.mean(dim=0)]

    # 0-D tensors for global posterior moments
    mu_global     = mu.clone().detach()
    sigma2_global = sigma2.clone().detach()
    pi_global     = float(pi)

    return mu_global, sigma2_global, pi_global, flags_nested, global_grad_scalar


def _sample_pi(flags: torch.Tensor, *, alpha0: float = 1.0, beta0: float = 1.0) -> float:
    n_total = flags.numel()
    n_one   = int(flags.sum().item())
    a_post  = alpha0 + n_one
    b_post  = beta0 + n_total - n_one
    return torch.distributions.Beta(float(a_post), float(b_post)).sample().item()


def _sample_flag(
    g: torch.Tensor,
    mu: torch.Tensor,
    sigma2: torch.Tensor,
    pi: float,
    *,
    outlier_pdf: float = 1e-3,
) -> torch.Tensor:
    std    = torch.sqrt(sigma2 + 1e-12)
    pdf_in = (1.0 / (std * math.sqrt(2.0 * math.pi))) * torch.exp(
        -(g - mu) ** 2 / (2.0 * sigma2 + 1e-12)
    )
    denom  = pi * outlier_pdf + (1.0 - pi) * pdf_in
    p_z1   = torch.where(
        denom == 0.0, torch.full_like(denom, 0.5),
        (pi * outlier_pdf) / denom
    )
    return torch.bernoulli(torch.clamp(p_z1, 0.0, 1.0))


def _sample_mean(
    g: torch.Tensor,
    z: torch.Tensor,
    *,
    tau0_sq: float = 1.0,
) -> torch.Tensor:
    honest = g[z == 0]
    n      = honest.numel()
    if n == 0:
        return torch.normal(
            torch.tensor(0.0, device=g.device, dtype=g.dtype),
            math.sqrt(tau0_sq),
        )

    m_hat   = honest.mean()
    var_hat = (
        honest.var(unbiased=True) if n > 1
        else torch.tensor(0.0, device=g.device, dtype=g.dtype)
    )
    post_var  = 1.0 / (1.0 / tau0_sq + n / (var_hat + 1e-12))
    post_mean = post_var * (m_hat / (var_hat + 1e-12))
    post_std  = torch.sqrt(post_var)
    return torch.normal(post_mean, post_std)


def _sample_variance(
    g: torch.Tensor,
    z: torch.Tensor,
    mu: torch.Tensor,
    *,
    alpha0: float = 1.0,
    beta0: float = 1.0,
) -> torch.Tensor:
    honest = g[z == 0]
    n      = honest.numel()
    if n == 0:
        return g.var(unbiased=True)

    sq_diff = (honest - mu) ** 2
    a_post  = alpha0 + n / 2.0
    b_post  = beta0  + 0.5 * sq_diff.sum()

    if g.device.type == "mps":               # MPS Gamma has to run on CPU
        concentration = torch.tensor(float(a_post), device="cpu", dtype=torch.float32)
        rate          = torch.tensor(float(b_post.item()), device="cpu", dtype=torch.float32)
        gamma_sample  = torch.distributions.Gamma(concentration, rate).sample()
        gamma_sample  = gamma_sample.to(g.device, dtype=g.dtype)
    else:
        gamma_sample = torch.distributions.Gamma(
            torch.tensor(a_post, device=g.device, dtype=g.dtype),
            torch.tensor(b_post, device=g.device, dtype=g.dtype),
        ).sample()

    return 1.0 / gamma_sample
