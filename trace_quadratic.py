"""
A toy example of a 9x9 matrix quadratic trace function optimization using stochastic gradient descent.
objective:
 f(X) = 1/2 Tr(X^T H X) + Tr(B^T X)
 X is a matrix variable, H is a symmetric positive semidefinite matrix, B is a matrix.
 we want to find the X that minimizes the objective.
 
this script runs the following optimizers:
 - SGD
 - Adam
 - Muon 
 - SOAP
 - DeVA$_\ell_\infty$
 - DeVA$_{S_\infty}$
 - Adamuon
 - SignSGD
"""

import os
import time
import json
import tyro
import torch
import numpy as np
import matplotlib.pyplot as plt

from optimizers.muon import SingleDeviceMuon
from optimizers.deva import (
    DeVAEuclideanNorm, DeVASchattenNorm, DeVASchattenNormSignStablization
)
from optimizers.adamuon import SingleDeviceAdamuon
from optimizers.signgd import SignSGD
from optimizers.soap import SOAP

from dataclasses import dataclass



# ----------------------------
# 1) Build H_hom / H_het exactly like earlier (3 blocks, Haar-ish rotations)
# ----------------------------
def random_rotation_haar_via_wishart(n=3, seed=0, device="cpu", dtype=torch.float64):
    g = torch.Generator(device=device); g.manual_seed(seed)
    A = torch.randn(n, n, generator=g, device=device, dtype=dtype)
    S = A @ A.T
    _, Q = torch.linalg.eigh(S)
    ## multiply Q by sign to remove randomness
    s = torch.sign(torch.diag(Q)); s[s == 0] = 1.0
    return Q * s

def build_block_diagonal_hessian(block_eigs, seed=0, device="cpu", dtype=torch.float64):
    H = torch.zeros(9, 9, device=device, dtype=dtype)
    for k, eigs in enumerate(block_eigs):
        Q = random_rotation_haar_via_wishart(3, seed=seed + 10*k, device=device, dtype=dtype)
        D = torch.diag(torch.tensor(eigs, device=device, dtype=dtype))
        B = Q @ D @ Q.T
        r0, r1 = 3*k, 3*(k+1)
        H[r0:r1, r0:r1] = B
    return 0.5 * (H + H.T)

def make_hom_het_hessians(seed=0, device="cpu", dtype=torch.float64):
    H_details_het = [[1,2,3], [99,100,101], [4998,4999,5000]]
    H_details_hom = [[1,99,4998], [2,100,4999], [3,101,5000]]
    H_het = build_block_diagonal_hessian(H_details_het, seed=seed, device=device, dtype=dtype)
    H_hom = build_block_diagonal_hessian(H_details_hom, seed=seed + 1000, device=device, dtype=dtype)
    return H_hom, H_het

def sqrt_psd(H):
    w, Q = torch.linalg.eigh(0.5*(H + H.T))
    w = torch.clamp(w, min=0)
    ### perform cholesky decomposition
    # L = torch.linalg.cholesky(H)
    return Q @ torch.diag(torch.sqrt(w)) @ Q.T

# ----------------------------
# 2) Matrix objective + metrics
# f(X) = 1/2 Tr(X^T H X) + Tr(B^T X)
# ----------------------------
def f_full_matrix(X, H, B):
    return 0.5 * torch.sum(X * (H @ X)) + torch.sum(B * X)  # scalar

def X_star_and_f_star(H, B):
    # Solve H X* = -B  (columnwise)
    X_star = torch.linalg.solve(H, -B)
    f_star = f_full_matrix(X_star, H, B)
    return X_star, f_star

# ----------------------------
# 3) Convert to least squares: 1/2 ||A X - Y||_F^2 + const
# where A = H^{1/2}, Y = -A^{-T} B
# ----------------------------
def make_A_Y_from_H_B(H, B):
    A = sqrt_psd(H)
    Y = -torch.linalg.solve(A.T, B)  # A^{-T} B
    return A, Y

def batch_loss_matrix(Xvar, A, Y, batch_size, g):
    # unbiased gradient estimator by scaling n/bs
    n = A.shape[0]
    idx = torch.randint(0, n, (batch_size,), generator=g, device=A.device)
    Ab = A[idx, :]     # (bs, d)
    Yb = Y[idx, :]     # (bs, p)
    scale = n / batch_size
    R = Ab @ Xvar - Yb  # (bs, p)
    return 0.5 * scale * torch.norm(R, p="fro")


def lr_schedule(
        step,
        total_steps,
        warmup_steps=100
):
    if step < warmup_steps:
        return 1.0
    
    # linear decay from 1.0 at step=warmup_steps to 0.0 at step=total_steps
    denom = max(1, total_steps - warmup_steps)
    t = (step - warmup_steps) / denom  # 0 -> 1
    return max(0.0, 1.0 - t)

# ----------------------------
# 4) Run SGD/Adam to optimize matrix variable Xvar
# ----------------------------
def run_optimizer_matrix(
    optimizer_name,
    H, B,
    p=8,                 # number of columns in matrix variable
    steps=6000,
    lr=1e-2,
    batch_size=3,
    seed=0,
    betas=(0.0,0.99),
):
    now = time.time()
    device, dtype = H.device, H.dtype
    d = H.shape[0]

    A, Y = make_A_Y_from_H_B(H, B)
    X_star, f_star = X_star_and_f_star(H, B)

    # same init across optimizers if you reuse seed externally
    g_init = torch.Generator(device=device); g_init.manual_seed(seed + 999)
    X0 = torch.randn(d, p, generator=g_init, device=device, dtype=dtype)

    # ----------------------------
    # Deterministic full-batch GD (no sampling, no torch.optim)
    # ----------------------------
    if optimizer_name in ["gd", "fullgd", "detgd"]:
        X = X0.clone()
        subopts = []
        for t in range(steps):
            # lr_t = lr * lr_schedule(t, total_steps=steps, warmup_steps=steps // 2)
            lr_t = lr
            # full deterministic gradient of f(X) = 1/2 Tr(X^T H X) + Tr(B^T X)
            grad = H @ X + B
            X = X - lr_t * grad

            with torch.no_grad():
                f_now = f_full_matrix(X, H, B)
                subopts.append((f_now - f_star).item())

        elarpes = time.time() - now
        return {"subopt": subopts, "elarpse": elarpes}

    Xvar = torch.nn.Parameter(X0.clone())

    if optimizer_name.lower() == "sgd":
        opt = torch.optim.SGD([Xvar], momentum=betas[0], lr=lr)
    elif optimizer_name.lower() == "signsgd":
        opt = SignSGD([Xvar], lr=lr, beta=betas[0])
    elif optimizer_name.lower() == "adam":
        opt = torch.optim.Adam([Xvar], betas=betas, lr=lr)
    elif optimizer_name.lower() == "adamuon":
        opt = SingleDeviceAdamuon([Xvar], betas=betas, weight_decay=0.0, lr=lr)
    elif optimizer_name.lower() == "muon":
        opt = SingleDeviceMuon([Xvar], momentum=betas[0], weight_decay=0.0, lr=lr)
    elif optimizer_name.lower() == "deva_sinfty":
        opt = DeVASchattenNorm([Xvar], betas=betas, weight_decay=0.0, lr=lr, precondition_frequency=10)
    elif optimizer_name.lower() == "deva_sinfty_sign":
        opt = DeVASchattenNormSignStablization([Xvar], betas=betas, weight_decay=0.0, lr=lr, precondition_frequency=10)
    elif optimizer_name.lower() == "deva_linfty":
        opt = DeVAEuclideanNorm([Xvar],betas=betas, weight_decay=0.0, lr=lr)
    elif optimizer_name.lower() == "soap":
        opt = SOAP(params=[Xvar], lr=lr, betas=betas, weight_decay=0.0, precondition_frequency=10)
    else:
        raise ValueError("optimizer_name must be 'sgd' or 'adam'")

    g = torch.Generator(device=device); g.manual_seed(seed + 12345)

    subopts = []
    distF = []
    gradF = []
    L_norm = []
    H_norm = []
    gamma = []

    for t in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = batch_loss_matrix(Xvar, A, Y, batch_size=batch_size, g=g)
        loss.backward()
        for pg in opt.param_groups:
            pg['lr'] = lr * lr_schedule(t, total_steps=steps, warmup_steps=steps // 2)
        opt.step()

        if optimizer_name in ["deva_sinfty", "deva_linfty"]:
            for pg in opt.param_groups:
                for p in pg["params"]:
                    state = opt.state[p]
                    L_norm.append(torch.sum(H *state["gamma"]).item())
                    H_norm.append(torch.sum(H).item())
                    gamma.append(state["gamma"].detach().cpu().numpy().tolist())

        with torch.no_grad():
            f_now = f_full_matrix(Xvar, H, B)
            subopts.append((f_now - f_star).item())
            # distF.append(torch.norm(Xvar - X_star).item())
            # grad = H @ Xvar + B
            # gradF.append(torch.norm(grad).item())
    elarpes = time.time() - now
    return {"subopt": subopts, "distF": distF, "gradF": gradF, "elarpse": elarpes, "L_norm": L_norm, "H_norm": H_norm, "gamma": gamma}


def plot_logmag(H, title=None, ax=None, eps=1e-12, add_colorbar=False, cbar_kw=None):
    """
    Plot log10(|H|+eps) as an image with proper colorbar alignment for insets.
    """
    if cbar_kw is None:
        cbar_kw = {}

    Hnp = H.detach().cpu().numpy() if hasattr(H, "detach") else np.asarray(H)
    img = np.log10(np.abs(Hnp) + eps)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    
    im = ax.imshow(img, cmap="coolwarm")
    ax.set_adjustable("box") # This is the "magic" line

    if title is not None:
        ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])

    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    if add_colorbar and ax.figure is not None:
        # Create a dedicated colorbar axes (cax) relative to the current 'ax'
        # This pins the colorbar to the right of the inset
        cax = inset_axes(
            ax,
            width="5%",             # Width of the colorbar
            height="100%",          # Match height of the image
            loc='lower left',
            bbox_to_anchor=(1.05, 0., 1, 1), # Offset slightly to the right (1.05)
            bbox_transform=ax.transAxes,
            borderpad=0
        )
        # Explicitly use cax so Matplotlib doesn't 'steal' space from ax
        cb = ax.figure.colorbar(im, cax=cax, **cbar_kw)
        cb.ax.tick_params(labelsize=7) # Small labels for insets

    return ax, im


def log_mean_band(y, eps=1e-30):
    y = np.asarray(y)
    y = np.where(np.isfinite(y) & (y > 0), y, np.nan)
    ly = np.log(np.maximum(y, eps))

    m = np.exp(np.nanmean(ly, axis=0))
    s = np.nanstd(ly, axis=0)

    lower = m * np.exp(-s)
    upper = m * np.exp(+s)
    return m, lower, upper

@dataclass
class Args:
    num_seeds: int = 10
    base_seed: int = 1234
    steps: int = 400
    batch_size: int = 3
    lr_signsgd: float = 0.1
    lr_muon: float = 0.1
    lr_adamuon: float = 0.1
    lr_gd: float = 0.0001
    beta1: float = 0.9
    beta2: float = 0.99

if __name__ == "__main__":

    args = tyro.cli(Args)
    base_seed = args.base_seed
    num_seeds = args.num_seeds
    steps = args.steps
    batch_size = args.batch_size
    lr_signsgd = args.lr_signsgd
    lr_muon = args.lr_muon
    lr_adamuon = args.lr_adamuon
    lr_gd = args.lr_gd
    beta1 = args.beta1
    beta2 = args.beta2
    p = 9
    seeds = [base_seed + i for i in range(num_seeds)]

    device = "cpu"
    dtype = torch.float32

    for i, heter in enumerate([True, False]):
        # -----------------------------
        # Optional: fix B across ALL seeds (so only H + optimizer randomness changes)
        # -----------------------------
        gB = torch.Generator(device=device)
        gB.manual_seed(base_seed + 2024)
        # B = torch.randn(9, p, generator=gB, device=device, dtype=dtype)
        B = torch.zeros(size=(9, p), device=device, dtype=dtype)

        # -----------------------------
        # Collect results across seeds
        # -----------------------------
        opt_specs = [
            ("gd",        dict(lr=lr_gd)),
            ("signsgd",   dict(lr=lr_signsgd)),
            ("adam",      dict(lr=lr_muon)),
            ("muon",      dict(lr=lr_muon)),
            ("soap",      dict(lr=lr_muon)),
            ("deva_sinfty",  dict(lr=lr_muon)),
            # ("deva_sinfty_sign",  dict(lr=lr_muon)),
            ("deva_linfty",  dict(lr=lr_muon)),
            ("adamuon",   dict(lr=lr_adamuon)),
        ]

        all_subopt = {name: [] for (name, _) in opt_specs}
        all_time   = {name: [] for (name, _) in opt_specs}
        all_L_norm = {name: [] for (name, _) in opt_specs}
        all_H_norm = {name: [] for (name, _) in opt_specs}
        all_gamma = {name: [] for (name, _) in opt_specs}
        for si, seed in enumerate(seeds):
            H_hom, H_het = make_hom_het_hessians(seed=seed, device=device, dtype=dtype)
            H = H_het if heter else H_hom


            # Run each optimizer on this seed
            for name, cfg in opt_specs:
                res = run_optimizer_matrix(
                    name, H, B, p=p, steps=steps, lr=cfg["lr"],
                    batch_size=batch_size, seed=seed + 10, betas=(beta1, beta2)
                )
                all_subopt[name].append(res["subopt"])
                all_time[name].append(res.get("elarpse", np.nan))
                all_L_norm[name].append(res.get("L_norm", np.nan))
                all_H_norm[name].append(res.get("H_norm", np.nan))
                all_gamma[name].append(res.get("gamma", np.nan))
        res = {
            "subopt": all_subopt, 
            "time": all_time, 
            "L_norm": all_L_norm, 
            "H_norm": all_H_norm, 
            "H": H.detach().cpu().numpy().tolist(),
            "gamma": all_gamma,
        }
        out_dir = f"results/"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"trace_quadratic_{num_seeds}seeds_{heter}_beta1_{beta1}_beta2_{beta2}.json"), "w") as f:
            json.dump(res, f)
        