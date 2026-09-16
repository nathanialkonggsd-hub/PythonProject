"""
PINN-Inverse-Darcy
==================
Physics-Informed Neural Network for 1D Inverse Darcy Flow.

Given sparse noisy observations of the hydraulic head h(x), infer the
spatially varying permeability field k(x).

Governing equation
------------------
    -d/dx ( k(x) * dh/dx ) = f(x),      x in [0, 1]

Boundary conditions
-------------------
    h(0) = 1,   h(1) = 0

Reference (ground truth)
------------------------
    k_true(x) = 1 + 0.5 * sin(2*pi*x)

    h_true(x) is obtained by solving the forward problem with a
    second-order finite-difference scheme, so that the synthetic
    observations are physically consistent with k_true.

Usage
-----
    python pinn_inverse_darcy.py                # default: 20000 epochs
    python pinn_inverse_darcy.py --epochs 40000
    python pinn_inverse_darcy.py --quick        # smoke test, 2000 epochs

Output
------
    figures/darcy_setup.png
    figures/inverse_training_loss.png
    figures/inverse_results.png
    figures/inversion_convergence.png

Repo: https://github.com/your-repo/PINN-Inverse-Darcy
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')          # headless backend, safe for server / CI
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Global plotting style
# ------------------------------------------------------------------
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.autolayout'] = True

OUTPUT_DIR = 'figures'


# ==================================================================
# 1. Configuration
# ==================================================================
class Config:
    """All problem and training hyperparameters."""

    # --- problem ---
    source = 1.0                    # constant source term f(x)

    # --- network ---
    n_hidden_h = 64                 # width of the h-network
    n_hidden_k = 64                 # width of the k-network
    n_layers_h = 4
    n_layers_k = 3

    # --- sampling ---
    n_col = 200                     # collocation points
    n_obs = 30                      # observation points
    obs_noise = 0.005               # 0.5% Gaussian noise on observations

    # --- optimization ---
    epochs = 20000
    lr = 1e-3
    lambda_data = 100.0             # weight of the data-fitting loss
    lambda_reg = 1e-6               # weight of the dk/dx regularization

    # --- k(x) parameterization bounds ---
    k_min = 0.3
    k_max = 1.8

    # --- reproducibility ---
    seed = 42
    # --- flux observations ---
    n_flux = 3                      # number of flux observations
    lambda_flux = 50.0              # weight of the flux loss


# ==================================================================
# 2. Reference permeability field
# ==================================================================
def reference_k(x):
    """Ground-truth permeability field."""
    return 1.0 + 0.5 * np.sin(2.0 * np.pi * x)


# ==================================================================
# 3. Forward solver (finite difference)
# ==================================================================
def solve_forward_fd(k_func, f_value, x_grid,
                     h_left=1.0, h_right=0.0):
    """
    Solve  -d/dx ( k(x) dh/dx ) = f  with Dirichlet BCs
    using a second-order finite-difference scheme.

    Why do we need this?
    --------------------
    The synthetic observations h_obs must be physically consistent
    with k_true. If h_true is an arbitrary analytic function, there
    exists no k(x) that simultaneously satisfies the PDE and the
    observations, and the PINN will collapse to a constant k.

    Parameters
    ----------
    k_func  : callable, k(x)
    f_value : float, constant source term
    x_grid  : 1D np.ndarray, uniform grid including endpoints
    h_left, h_right : Dirichlet boundary values

    Returns
    -------
    h : 1D np.ndarray, hydraulic head on x_grid
    """
    n = len(x_grid)
    dx = x_grid[1] - x_grid[0]

    # k at cell centers (midpoints between grid nodes)
    k_half = k_func(0.5 * (x_grid[:-1] + x_grid[1:]))   # length n-1

    # Assemble tridiagonal system A h = b
    A = np.zeros((n, n))
    b = np.full(n, f_value * dx * dx)

    for i in range(1, n - 1):
        kL = k_half[i - 1]      # k at (i - 1/2)
        kR = k_half[i]          # k at (i + 1/2)
        A[i, i - 1] = -kL
        A[i, i]     =  kL + kR
        A[i, i + 1] = -kR

    # Dirichlet BCs
    A[0, 0]     = 1.0
    b[0]        = h_left
    A[-1, -1]   = 1.0
    b[-1]       = h_right

    h = np.linalg.solve(A, b)
    return h


# ==================================================================
# 4. Data generation
# ==================================================================
def generate_data(cfg):
    rng = np.random.default_rng(cfg.seed)

    # ---- 1. forward solve ----
    x_fine = np.linspace(0.0, 1.0, 401)
    k_fine = reference_k(x_fine)
    h_fine = solve_forward_fd(reference_k, cfg.source, x_fine,
                              h_left=1.0, h_right=0.0)

    # ---- 2. head observations ----
    x_obs = np.linspace(0.05, 0.95, cfg.n_obs)
    h_obs = np.interp(x_obs, x_fine, h_fine)
    h_obs = h_obs + cfg.obs_noise * rng.standard_normal(cfg.n_obs)

    # ---- 3. flux observations ----
    # q(x) = -k(x) * dh/dx
    # We compute h' by finite difference on the fine grid, then
    # evaluate q at n_flux uniformly spaced points.
    h_prime_fine = np.gradient(h_fine, x_fine)
    q_fine = -k_fine * h_prime_fine
    x_flux = np.linspace(0.1, 0.9, cfg.n_flux)
    q_obs = np.interp(x_flux, x_fine, q_fine)
    q_obs = q_obs + cfg.obs_noise * rng.standard_normal(cfg.n_flux)

    # ---- 4. collocation & test grids ----
    x_col = np.linspace(0.0, 1.0, cfg.n_col)
    x_test = np.linspace(0.0, 1.0, 400)

    to_tensor = lambda a: torch.tensor(a, dtype=torch.float32).unsqueeze(1)

    return {
        'x_col':  to_tensor(x_col),
        'x_obs':  to_tensor(x_obs),
        'h_obs':  to_tensor(h_obs),
        'x_flux': to_tensor(x_flux),
        'q_obs':  to_tensor(q_obs),
        'x_test': to_tensor(x_test),
        'x_fine': x_fine,
        'h_fine': h_fine,
        'k_fine': k_fine,
    }


# ==================================================================
# 5. Networks
# ==================================================================
class MLP(nn.Module):
    """Tanh MLP with a trainable activation slope."""

    def __init__(self, in_dim, out_dim, hidden, n_layers):
        super().__init__()
        dims = [in_dim] + [hidden] * n_layers + [out_dim]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        for lin in self.linears[:-1]:
            x = torch.tanh(self.alpha * lin(x))
        return self.linears[-1](x)


class InverseDarcyPINN(nn.Module):
    """
    Two sub-networks:
      - h_net : maps x -> hydraulic head h(x)
      - k_net : maps x -> permeability k(x), bounded to [k_min, k_max]
    """

    def __init__(self, cfg):
        super().__init__()
        self.h_net = MLP(1, 1, cfg.n_hidden_h, cfg.n_layers_h)
        self.k_net = MLP(1, 1, cfg.n_hidden_k, cfg.n_layers_k)
        self.k_min = cfg.k_min
        self.k_max = cfg.k_max

    def forward_h(self, x):
        return self.h_net(x)

    def forward_k(self, x):
        # Bounded parameterization: k in [k_min, k_max].
        # Why?
        # A softplus output is unbounded above. When the regularization
        # term pushes k toward a constant, the network is free to pick
        # any constant value. A tanh-based bounded output constrains the
        # "flattening" to a fixed interval, which helps the shape of k(x)
        # survive training.
        raw = torch.tanh(self.k_net(x))                     # in (-1, 1)
        return self.k_min + 0.5 * (self.k_max - self.k_min) * (raw + 1.0)


# ==================================================================
# 6. PDE residual
# ==================================================================
def pde_residual(model, x, source=1.0):
    """
    Compute the residual of
        -d/dx ( k(x) * dh/dx ) - f(x)
    """
    x = x.clone().requires_grad_(True)
    h = model.forward_h(x)
    k = model.forward_k(x)

    h_x = torch.autograd.grad(h, x, torch.ones_like(h), create_graph=True)[0]
    k_x = torch.autograd.grad(k, x, torch.ones_like(k), create_graph=True)[0]

    # d/dx (k * h_x) = k_x * h_x + k * h_xx
    flux = k * h_x
    flux_x = torch.autograd.grad(flux, x, torch.ones_like(flux),
                                 create_graph=True)[0]

    return -flux_x - source


# ==================================================================
# 7. Training
# ==================================================================
def train(model, data, cfg, device='cpu'):
    """Full training loop. Returns a dict of per-epoch loss history."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5
    )

    history = {
        'pde': [], 'data': [], 'reg': [], 'flux': [], 'total': [],
        'k_err': [], 'k_err_epoch': [],
    }

    x_col = data['x_col'].to(device)
    x_obs = data['x_obs'].to(device)
    h_obs = data['h_obs'].to(device)

    # Pre-compute ground-truth k on collocation points (for diagnostics only;
    # the training loss never sees this).
    k_true_col = torch.tensor(
        reference_k(x_col.cpu().numpy().flatten()),
        dtype=torch.float32, device=device,
    ).unsqueeze(1)

    log_every = max(1, cfg.epochs // 10)
    diag_every = max(1, cfg.epochs // 300)

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()

        # (a) PDE residual loss
        f = pde_residual(model, x_col, cfg.source)
        loss_pde = torch.mean(f ** 2)

        # (b) data-fitting loss
        h_pred = model.forward_h(x_obs)
        loss_data = torch.mean((h_pred - h_obs) ** 2)

        # (c) regularization on dk/dx, suppresses spurious oscillations
        x_reg = x_col.clone().requires_grad_(True)
        k_reg = model.forward_k(x_reg)
        k_x = torch.autograd.grad(k_reg, x_reg,
                                  torch.ones_like(k_reg),
                                  create_graph=True)[0]
        loss_reg = torch.mean(k_x ** 2)

        # (d) flux loss: q_pred = -k * dh/dx
        x_flux = data['x_flux'].to(device)
        q_obs = data['q_obs'].to(device)

        x_flux_req = x_flux.clone().requires_grad_(True)
        h_flux = model.forward_h(x_flux_req)
        k_flux = model.forward_k(x_flux_req)
        h_x_flux = torch.autograd.grad(
            h_flux, x_flux_req,
            torch.ones_like(h_flux), create_graph=True
        )[0]
        q_pred = -k_flux * h_x_flux
        loss_flux = torch.mean((q_pred - q_obs) ** 2)

        loss = (loss_pde
                + cfg.lambda_data * loss_data
                + cfg.lambda_reg * loss_reg
                + cfg.lambda_flux * loss_flux)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        history['pde'].append(loss_pde.item())
        history['data'].append(loss_data.item())
        history['reg'].append(loss_reg.item())
        history['total'].append(loss.item())
        history['flux'].append(loss_flux.item())

        # diagnostic: relative L2 error of k(x) on collocation points
        if epoch % diag_every == 0:
            with torch.no_grad():
                k_pred_col = model.forward_k(x_col)
            rel = (torch.norm(k_pred_col - k_true_col)
                   / torch.norm(k_true_col)).item()
            history['k_err'].append(rel)
            history['k_err_epoch'].append(epoch)

        if epoch % log_every == 0:
            print(f"  [{epoch:5d}/{cfg.epochs}] "
                  f"pde={loss_pde.item():.3e}  "
                  f"data={loss_data.item():.3e}  "
                  f"reg={loss_reg.item():.3e}")

    return history


# ==================================================================
# 8. Evaluation
# ==================================================================
@torch.no_grad()
def evaluate(model, data, device='cpu'):
    """Return predictions on the test grid together with relative errors."""
    model.eval()
    x_test = data['x_test'].to(device)

    h_pred = model.forward_h(x_test).cpu().numpy().flatten()
    k_pred = model.forward_k(x_test).cpu().numpy().flatten()

    x_np = data['x_test'].cpu().numpy().flatten()

    # interpolate the FD solution onto the test grid
    h_true = np.interp(x_np, data['x_fine'], data['h_fine'])
    k_true = reference_k(x_np)

    rel_err_h = np.linalg.norm(h_pred - h_true) / np.linalg.norm(h_true)
    rel_err_k = np.linalg.norm(k_pred - k_true) / np.linalg.norm(k_true)

    return {
        'x': x_np,
        'h_pred': h_pred, 'h_true': h_true,
        'k_pred': k_pred, 'k_true': k_true,
        'rel_err_h': rel_err_h, 'rel_err_k': rel_err_k,
    }


# ==================================================================
# 9. Plotting
# ==================================================================
def _ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def plot_domain_setup(cfg, outdir):
    """Figure 1: problem setup for the forward and inverse problems."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # --- forward problem ---
    ax = axes[0]
    ax.plot([0, 1], [0, 0], 'k-', linewidth=3)
    x_col = np.linspace(0.02, 0.98, 30)
    ax.scatter(x_col, np.zeros_like(x_col), s=15, c='gray', alpha=0.6,
               zorder=5, label='Collocation points')
    ax.scatter([0, 1], [0, 0], s=120, c='blue', marker='s', zorder=6)
    ax.annotate('h(0)=1', xy=(0, 0), xytext=(-0.08, 0.12),
                fontsize=12, color='blue', fontweight='bold')
    ax.annotate('h(1)=0', xy=(1, 0), xytext=(1.02, 0.12),
                fontsize=12, color='blue', fontweight='bold')
    ax.text(0.5, -0.30,
            r'$-\frac{d}{dx}\left(k(x)\frac{dh}{dx}\right) = f(x)$'
            '\n' r'Given: $k(x) = 1 + 0.5\sin(2\pi x)$',
            ha='center', fontsize=13)
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.5, 0.35)
    ax.axis('off')
    ax.set_title('Forward Problem: given k(x), solve h(x)',
                 fontsize=13, pad=10)

    # --- inverse problem ---
    ax = axes[1]
    ax.plot([0, 1], [0, 0], 'k-', linewidth=3)
    x_obs = np.linspace(0.05, 0.95, 10)
    ax.scatter(x_obs, np.zeros_like(x_obs), s=60, c='red',
               marker='o', zorder=6, label='Observations h')
    ax.scatter([0, 1], [0, 0], s=120, c='blue', marker='s', zorder=6)
    ax.annotate('h(0)=1', xy=(0, 0), xytext=(-0.08, 0.12),
                fontsize=12, color='blue', fontweight='bold')
    ax.annotate('h(1)=0', xy=(1, 0), xytext=(1.02, 0.12),
                fontsize=12, color='blue', fontweight='bold')
    ax.text(0.5, -0.30,
            r'$-\frac{d}{dx}\left(k(x)\frac{dh}{dx}\right) = f(x)$'
            '\n' r'Unknown: $k(x)$ (spatially varying field)',
            ha='center', fontsize=13, color='red')
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.5, 0.35)
    ax.axis('off')
    ax.set_title('Inverse Problem: given sparse h, infer k(x)',
                 fontsize=13, pad=10)

    path = os.path.join(outdir, 'darcy_setup.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


def plot_training_loss(history, outdir):
    """Figure 2: per-epoch component and total losses."""
    epochs = np.arange(len(history['pde']))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(epochs, history['pde'],  'b-', label=r'$L_{PDE}$',  lw=2)
    axes[0].semilogy(epochs, history['data'], 'r-', label=r'$L_{data}$', lw=2)
    axes[0].semilogy(epochs, history['reg'],  'g-', label=r'$L_{reg}$',  lw=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_title('Component Losses', fontsize=13, pad=10)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs, history['total'], 'k-', lw=2, label='Total')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Total Loss (log scale)')
    axes[1].set_title('Total Loss', fontsize=13, pad=10)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    path = os.path.join(outdir, 'inverse_training_loss.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


def plot_inverse_results(results, obs_data, outdir):
    """Figure 3: h(x) prediction and k(x) inversion."""
    x = results['x']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(x, results['h_true'], 'k--', lw=2, label='True h(x)')
    axes[0].plot(x, results['h_pred'], 'b-',  lw=2, label='PINN prediction')
    axes[0].scatter(obs_data['x_obs'], obs_data['h_obs'], c='red', s=60,
                    zorder=5, label='Observations')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('h(x)')
    axes[0].set_title(
        f"Hydraulic Head  (rel. err = {results['rel_err_h']*100:.2f}%)",
        fontsize=13, pad=10,
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, results['k_true'], 'k--', lw=2, label='True k(x)')
    axes[1].plot(x, results['k_pred'], 'r-',  lw=2, label='Inverted k(x)')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('k(x)')
    axes[1].set_title(
        f"Permeability Inversion  (rel. err = {results['rel_err_k']*100:.2f}%)",
        fontsize=13, pad=10,
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    path = os.path.join(outdir, 'inverse_results.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


def plot_convergence(history, outdir):
    """Figure 4: relative error of k(x) vs. training epoch."""
    epochs = np.array(history['k_err_epoch'])
    k_err = np.array(history['k_err']) * 100.0     # to percent

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(epochs, k_err, 'b-', lw=2, label='Relative error of k(x)')
    ax.axhline(y=5, color='r', linestyle='--', lw=1.5, label='5% threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Relative error (%)')
    ax.set_title('Inversion Error Convergence', fontsize=13, pad=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, 'inversion_convergence.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


# ==================================================================
# 10. Main
# ==================================================================
def main():
    parser = argparse.ArgumentParser(
        description='PINN for 1D inverse Darcy flow.'
    )
    parser.add_argument('--epochs', type=int, default=Config.epochs,
                        help='number of training epochs')
    parser.add_argument('--quick', action='store_true',
                        help='smoke test: 2000 epochs')
    args = parser.parse_args()

    cfg = Config()
    cfg.epochs = 2000 if args.quick else args.epochs

    # reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    outdir = _ensure_outdir()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 60)
    print('PINN-Inverse-Darcy')
    print(f'  device  : {device}')
    print(f'  epochs  : {cfg.epochs}')
    print(f'  output  : {outdir}/')
    print('=' * 60)

    # 1. data
    data = generate_data(cfg)

    # 2. model
    model = InverseDarcyPINN(cfg)

    # 3. train
    print('\n[Training]')
    history = train(model, data, cfg, device=device)

    # 4. evaluate
    print('\n[Evaluation]')
    results = evaluate(model, data, device=device)
    print(f"  relative L2 error of h : {results['rel_err_h']*100:.3f}%")
    print(f"  relative L2 error of k : {results['rel_err_k']*100:.3f}%")

    # 5. plots
    print('\n[Plots]')
    obs_data = {
        'x_obs': data['x_obs'].cpu().numpy().flatten(),
        'h_obs': data['h_obs'].cpu().numpy().flatten(),
    }
    plot_domain_setup(cfg, outdir)
    plot_training_loss(history, outdir)
    plot_inverse_results(results, obs_data, outdir)
    plot_convergence(history, outdir)

    print('\nDone.')


if __name__ == '__main__':
    main()