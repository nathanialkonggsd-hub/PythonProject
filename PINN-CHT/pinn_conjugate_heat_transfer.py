"""
PINN-CHT: Physics-Informed Neural Network for 2D Steady Conjugate Heat Transfer

Problem
-------
Domain: [0, 1] x [0, 1]
    - Solid    : y in [0, 0.5]
    - Fluid    : y in [0.5, 1]
    - Interface: y = 0.5

Governing equations
-------------------
Solid:
    k_s * (T_xx + T_yy) = 0
Fluid (with prescribed velocity u(y), v = 0):
    u * T_x + v * T_y = alpha_f * (T_xx + T_yy)

Interface conditions at y = 0.5:
    T_s = T_f
    k_s * dT_s/dy = k_f * dT_f/dy

Boundary conditions
-------------------
    T_s(x, 0) = 1.0                     (heated bottom)
    T_f(x, 1) = 0.0                     (cooled top)
    T_f(0, y) = 4*(y-0.5)*(1-y)         (smooth inlet profile,
                                         vanishes at y=0.5 and y=1)

Two coupling schemes
--------------------
    single : one shared MLP, domain flag as input
    dual   : two separate MLPs, interface loss for coupling

Usage
-----
    python pinn_conjugate_heat_transfer.py --scheme dual
    python pinn_conjugate_heat_transfer.py --scheme single
    python pinn_conjugate_heat_transfer.py --scheme compare
    python pinn_conjugate_heat_transfer.py --quick
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    # --- physics ---
    k_s = 1.0                 # solid thermal conductivity
    k_f = 0.5                 # fluid thermal conductivity
    alpha_f = 0.1             # fluid thermal diffusivity
    y_interface = 0.5         # interface location

    # --- network ---
    n_hidden = 64
    n_layers = 4

    # --- sampling ---
    n_solid = 2000
    n_fluid = 3000
    n_interface = 200
    n_bc = 200
    n_inlet = 200             # inlet boundary sampling

    # --- optimization ---
    epochs = 25000
    lr = 1e-3
    lambda_interface = 100.0
    lambda_bc = 100.0

    # --- reproducibility ---
    seed = 42


# ==================================================================
# 2. Velocity field (prescribed parabolic profile)
# ==================================================================
def velocity_field(y):
    """Parabolic velocity profile in the fluid domain: u(y) = 4y(1-y)."""
    u = 4.0 * y * (1.0 - y)
    v = torch.zeros_like(u)
    return u, v


# ==================================================================
# 3. Networks
# ==================================================================
class MLP(nn.Module):
    """Tanh MLP with trainable activation slope."""

    def __init__(self, in_dim, out_dim, n_hidden=64, n_layers=4):
        super().__init__()
        dims = [in_dim] + [n_hidden] * n_layers + [out_dim]
        self.linears = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        for lin in self.linears[:-1]:
            x = torch.tanh(self.alpha * lin(x))
        return self.linears[-1](x)


class SingleNetCHT(nn.Module):
    """
    Scheme A: single network with domain flag.
    Input: (x, y, s) where s = 0 (solid) or 1 (fluid).
    Output: T(x, y)
    """

    def __init__(self, cfg):
        super().__init__()
        self.net = MLP(3, 1, cfg.n_hidden, cfg.n_layers)

    def forward(self, x, y, s):
        return self.net(torch.cat([x, y, s], dim=1))

    def predict_solid(self, x, y):
        return self.forward(x, y, torch.zeros_like(x))

    def predict_fluid(self, x, y):
        return self.forward(x, y, torch.ones_like(x))


class DualNetCHT(nn.Module):
    """
    Scheme B: two separate networks, one per physical domain.
    """

    def __init__(self, cfg):
        super().__init__()
        self.net_solid = MLP(2, 1, cfg.n_hidden, cfg.n_layers)
        self.net_fluid = MLP(2, 1, cfg.n_hidden, cfg.n_layers)

    def predict_solid(self, x, y):
        return self.net_solid(torch.cat([x, y], dim=1))

    def predict_fluid(self, x, y):
        return self.net_fluid(torch.cat([x, y], dim=1))


# ==================================================================
# 4. Residuals
# ==================================================================
def solid_residual(predict_fn, x, y, k_s):
    """Residual of the solid heat equation: k_s * (T_xx + T_yy)."""
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)

    T = predict_fn(x, y)
    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, torch.ones_like(T_y), create_graph=True)[0]

    return k_s * (T_xx + T_yy)


def fluid_residual(predict_fn, x, y, alpha_f):
    """Residual of the fluid energy equation: u*T_x + v*T_y - alpha_f*(T_xx + T_yy)."""
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)

    T = predict_fn(x, y)
    u, v = velocity_field(y)

    T_x = torch.autograd.grad(T, x, torch.ones_like(T), create_graph=True)[0]
    T_y = torch.autograd.grad(T, y, torch.ones_like(T), create_graph=True)[0]
    T_xx = torch.autograd.grad(T_x, x, torch.ones_like(T_x), create_graph=True)[0]
    T_yy = torch.autograd.grad(T_y, y, torch.ones_like(T_y), create_graph=True)[0]

    return u * T_x + v * T_y - alpha_f * (T_xx + T_yy)


def interface_loss(predict_solid, predict_fluid, x_int, y_int, k_s, k_f):
    """
    Interface coupling loss:
        1. Temperature continuity: T_s = T_f
        2. Heat flux continuity : k_s * dT_s/dy = k_f * dT_f/dy
    """
    x = x_int.clone().requires_grad_(True)
    y = y_int.clone().requires_grad_(True)

    T_s = predict_solid(x, y)
    T_f = predict_fluid(x, y)

    loss_temp = torch.mean((T_s - T_f) ** 2)

    T_s_y = torch.autograd.grad(T_s, y, torch.ones_like(T_s), create_graph=True)[0]
    T_f_y = torch.autograd.grad(T_f, y, torch.ones_like(T_f), create_graph=True)[0]
    loss_flux = torch.mean((k_s * T_s_y - k_f * T_f_y) ** 2)

    return loss_temp, loss_flux


# ==================================================================
# 5. Data generation
# ==================================================================
def generate_data(cfg):
    """Build collocation, interface, inlet and BC tensors."""
    rng = np.random.default_rng(cfg.seed)

    def to_t(a):
        return torch.tensor(a, dtype=torch.float32).unsqueeze(1)

    # solid domain
    x_solid = to_t(rng.uniform(0.0, 1.0, cfg.n_solid))
    y_solid = to_t(rng.uniform(0.0, cfg.y_interface, cfg.n_solid))

    # fluid domain
    x_fluid = to_t(rng.uniform(0.0, 1.0, cfg.n_fluid))
    y_fluid = to_t(rng.uniform(cfg.y_interface, 1.0, cfg.n_fluid))

    # interface
    x_int = to_t(rng.uniform(0.0, 1.0, cfg.n_interface))
    y_int = to_t(np.full(cfg.n_interface, cfg.y_interface))

    # bottom BC (solid, T = 1)
    x_bc_bot = to_t(rng.uniform(0.0, 1.0, cfg.n_bc))
    y_bc_bot = to_t(np.zeros(cfg.n_bc))

    # top BC (fluid, T = 0)
    x_bc_top = to_t(rng.uniform(0.0, 1.0, cfg.n_bc))
    y_bc_top = to_t(np.ones(cfg.n_bc))

    # inlet BC (fluid, x = 0):
    # Smooth profile that vanishes at both y=0.5 (interface) and y=1 (top wall),
    # so it is compatible with the adjacent boundary conditions.
    #   T(0, y) = 4 * (y - 0.5) * (1 - y)
    x_inlet = to_t(np.zeros(cfg.n_inlet))
    y_inlet = to_t(rng.uniform(cfg.y_interface, 1.0, cfg.n_inlet))
    T_inlet = 4.0 * (y_inlet - cfg.y_interface) * (1.0 - y_inlet)

    return {
        'x_solid': x_solid, 'y_solid': y_solid,
        'x_fluid': x_fluid, 'y_fluid': y_fluid,
        'x_int':   x_int,   'y_int':   y_int,
        'x_bc_bot': x_bc_bot, 'y_bc_bot': y_bc_bot,
        'x_bc_top': x_bc_top, 'y_bc_top': y_bc_top,
        'x_inlet': x_inlet, 'y_inlet': y_inlet, 'T_inlet': T_inlet,
    }


# ==================================================================
# 6. Training
# ==================================================================
def train(model, data, cfg, device='cpu'):
    """Full training loop. Returns a dict of per-epoch loss history."""
    model.to(device)
    data = {k: v.to(device) for k, v in data.items()}

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.5
    )

    history = {
        'solid': [], 'fluid': [], 'interface': [],
        'bc': [], 'inlet': [], 'total': [],
    }

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()

        # (a) solid PDE residual
        f_s = solid_residual(
            model.predict_solid, data['x_solid'], data['y_solid'], cfg.k_s
        )
        loss_solid = torch.mean(f_s ** 2)

        # (b) fluid PDE residual
        f_f = fluid_residual(
            model.predict_fluid, data['x_fluid'], data['y_fluid'], cfg.alpha_f
        )
        loss_fluid = torch.mean(f_f ** 2)

        # (c) interface coupling
        loss_temp, loss_flux = interface_loss(
            model.predict_solid, model.predict_fluid,
            data['x_int'], data['y_int'], cfg.k_s, cfg.k_f
        )
        loss_interface = loss_temp + loss_flux

        # (d) bottom and top boundary conditions
        T_bot = model.predict_solid(data['x_bc_bot'], data['y_bc_bot'])
        T_top = model.predict_fluid(data['x_bc_top'], data['y_bc_top'])
        loss_bc = torch.mean((T_bot - 1.0) ** 2) + torch.mean(T_top ** 2)

        # (e) inlet boundary condition
        T_in = model.predict_fluid(data['x_inlet'], data['y_inlet'])
        loss_inlet = torch.mean((T_in - data['T_inlet']) ** 2)

        # (f) weighted total loss
        loss = (loss_solid + loss_fluid
                + cfg.lambda_interface * loss_interface
                + cfg.lambda_bc * loss_bc
                + cfg.lambda_bc * loss_inlet)

        loss.backward()
        optimizer.step()
        scheduler.step()

        history['solid'].append(loss_solid.item())
        history['fluid'].append(loss_fluid.item())
        history['interface'].append(loss_interface.item())
        history['bc'].append(loss_bc.item())
        history['inlet'].append(loss_inlet.item())
        history['total'].append(loss.item())

        if epoch % max(1, cfg.epochs // 10) == 0:
            print(f"  [{epoch:5d}/{cfg.epochs}] "
                  f"solid={loss_solid.item():.2e}  "
                  f"fluid={loss_fluid.item():.2e}  "
                  f"intf={loss_interface.item():.2e}  "
                  f"bc={loss_bc.item():.2e}  "
                  f"inlet={loss_inlet.item():.2e}")

    return history


# ==================================================================
# 7. Evaluation
# ==================================================================
@torch.no_grad()
def evaluate(model, cfg, n_grid=100):
    """Evaluate the temperature field on a regular grid."""
    x = torch.linspace(0.0, 1.0, n_grid)
    y = torch.linspace(0.0, 1.0, n_grid)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    X_flat = X.flatten().unsqueeze(1)
    Y_flat = Y.flatten().unsqueeze(1)

    T_s = model.predict_solid(X_flat, Y_flat).flatten()
    T_f = model.predict_fluid(X_flat, Y_flat).flatten()

    mask_solid = (Y_flat.flatten() <= cfg.y_interface)
    T_field = torch.where(mask_solid, T_s, T_f).reshape(n_grid, n_grid)

    return X.numpy(), Y.numpy(), T_field.numpy()


# ==================================================================
# 8. Plotting
# ==================================================================
def _ensure_outdir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def plot_setup(cfg, outdir):
    """Figure 1: problem setup."""
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.add_patch(plt.Rectangle((0, 0), 1, cfg.y_interface,
                               facecolor='lightblue', alpha=0.5))
    ax.add_patch(plt.Rectangle((0, cfg.y_interface), 1, 1 - cfg.y_interface,
                               facecolor='lightyellow', alpha=0.5))
    ax.axhline(y=cfg.y_interface, color='red', linewidth=2, linestyle='--')

    ax.text(0.5, 0.25, 'Solid  ($k_s$)', ha='center', fontsize=14)
    ax.text(0.5, 0.75, 'Fluid  ($k_f$, u(y))', ha='center', fontsize=14)
    ax.text(0.5, 0.51, 'Interface  $\\Gamma$', color='red',
            ha='center', fontsize=12)

    ax.annotate('T = 1.0', xy=(0.5, 0), xytext=(0.5, -0.09),
                ha='center', fontsize=12, color='blue', fontweight='bold')
    ax.annotate('T = 0.0', xy=(0.5, 1), xytext=(0.5, 1.03),
                ha='center', fontsize=12, color='blue', fontweight='bold')
    ax.annotate('T(0,y)=4(y-0.5)(1-y)',
                xy=(0, 0.75), xytext=(-0.45, 0.75),
                fontsize=10, color='green', fontweight='bold',
                rotation=90, va='center')

    ax.set_xlim(-0.55, 1.05)
    ax.set_ylim(-0.12, 1.1)
    ax.set_aspect('equal')
    ax.set_title('Conjugate Heat Transfer: Problem Setup', fontsize=13, pad=10)

    path = os.path.join(outdir, 'cht_setup.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


def plot_training_loss(history, outdir, tag=''):
    """Figure 2: per-epoch component and total losses."""
    epochs = np.arange(len(history['total']))
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.semilogy(epochs, history['solid'],     label='Solid PDE',  lw=1.5)
    ax.semilogy(epochs, history['fluid'],     label='Fluid PDE',  lw=1.5)
    ax.semilogy(epochs, history['interface'], label='Interface',  lw=1.5)
    ax.semilogy(epochs, history['bc'],        label='Top/Bottom BC', lw=1.5)
    ax.semilogy(epochs, history['inlet'],     label='Inlet BC',   lw=1.5)
    ax.semilogy(epochs, history['total'], 'k-', label='Total', lw=2)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title(f'Training Loss {tag}', fontsize=13, pad=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, f'cht_training_loss{tag}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


def plot_temperature(X, Y, T, outdir, tag=''):
    """Figure 3: temperature field."""
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(X, Y, T, levels=50, cmap='jet')
    plt.colorbar(cf, ax=ax, label='T(x, y)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Temperature Field {tag}', fontsize=13, pad=10)
    ax.set_aspect('equal')

    path = os.path.join(outdir, f'cht_temperature{tag}.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


def plot_comparison(hist_single, hist_dual, outdir):
    """Figure 4: single vs dual convergence comparison."""
    fig, ax = plt.subplots(figsize=(10, 5))

    epochs_s = np.arange(len(hist_single['total']))
    epochs_d = np.arange(len(hist_dual['total']))

    ax.semilogy(epochs_s, hist_single['total'], 'r-', lw=2,
                label='Single Net (Total)')
    ax.semilogy(epochs_d, hist_dual['total'], 'b-', lw=2,
                label='Dual Net (Total)')
    ax.semilogy(epochs_s, hist_single['interface'], 'r--', lw=1.5,
                label='Single (Interface)')
    ax.semilogy(epochs_d, hist_dual['interface'], 'b--', lw=1.5,
                label='Dual (Interface)')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Single vs Dual Network: Training Comparison',
                 fontsize=13, pad=10)
    ax.legend()
    ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, 'cht_comparison.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  saved: {path}')


# ==================================================================
# 9. Main
# ==================================================================
def main():
    parser = argparse.ArgumentParser(
        description='PINN for 2D steady conjugate heat transfer.'
    )
    parser.add_argument('--scheme', choices=['single', 'dual', 'compare'],
                        default='dual',
                        help='coupling scheme: single, dual, or compare both')
    parser.add_argument('--quick', action='store_true',
                        help='smoke test: 2000 epochs, reduced sampling')
    args = parser.parse_args()

    cfg = Config()
    if args.quick:
        cfg.epochs = 2000
        cfg.n_solid = 500
        cfg.n_fluid = 800
        cfg.n_interface = 50
        cfg.n_bc = 50
        cfg.n_inlet = 50

    outdir = _ensure_outdir()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('=' * 60)
    print('PINN-CHT')
    print(f'  scheme  : {args.scheme}')
    print(f'  device  : {device}')
    print(f'  epochs  : {cfg.epochs}')
    print(f'  output  : {outdir}/')
    print('=' * 60)

    data = generate_data(cfg)

    plot_setup(cfg, outdir)

    hist_single = None
    hist_dual = None

    if args.scheme in ('single', 'compare'):
        print('\n[Scheme A: Single Network]')
        torch.manual_seed(cfg.seed)
        model_s = SingleNetCHT(cfg)
        hist_single = train(model_s, data, cfg, device=device)

        plot_training_loss(hist_single, outdir, tag='_single')
        X, Y, T = evaluate(model_s, cfg)
        plot_temperature(X, Y, T, outdir, tag='_single')

    if args.scheme in ('dual', 'compare'):
        print('\n[Scheme B: Dual Network]')
        torch.manual_seed(cfg.seed)
        model_d = DualNetCHT(cfg)
        hist_dual = train(model_d, data, cfg, device=device)

        plot_training_loss(hist_dual, outdir, tag='_dual')
        X, Y, T = evaluate(model_d, cfg)
        plot_temperature(X, Y, T, outdir, tag='_dual')

    if args.scheme == 'compare' and hist_single is not None and hist_dual is not None:
        plot_comparison(hist_single, hist_dual, outdir)

    print('\nDone.')


if __name__ == '__main__':
    main()