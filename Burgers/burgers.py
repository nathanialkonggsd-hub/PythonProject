import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ============ 1. Configuration ============
NU = 0.01 / np.pi  # Viscosity coefficient
DOMAIN_X = (-1.0, 1.0)
DOMAIN_T = (0.0, 1.0)
N_COL = 5000  # Collocation points (reduced to 5000 for memory safety, can be increased to 10000)
N_IC = 200  # Initial condition points


# ============ 2. Model Definition (Hard Constraint + Wider Network) ============
class PINN(nn.Module):
    """
    Physics-Informed Neural Network with Hard Constraint for Burgers Equation.
    Network: [2, 128, 128, 128, 1]
    Hard constraint: u(-1, t) = 0 and u(1, t) = 0
    """

    def __init__(self, layers=[2, 128, 128, 128, 1]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        # Adaptive activation slope parameter (learnable)
        self.alpha = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x, t):
        inputs = torch.cat([x, t], dim=1)
        for i, layer in enumerate(self.layers[:-1]):
            inputs = layer(inputs)
            inputs = torch.tanh(self.alpha * inputs)
        raw = self.layers[-1](inputs)
        # Hard constraint: u(-1, t) = 0, u(1, t) = 0
        # (1 - x^2) is zero at x = -1 and x = 1
        u = (1.0 - x ** 2) * raw
        return u


# ============ 3. PDE Residual (Automatic Differentiation) ============
def compute_pde_residual(model, x, t, nu=0.01):
    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # f = u_t + u*u_x - nu*u_xx
    f = u_t + u * u_x - nu * u_xx
    return f


# ============ 4. Loss Function (No BC loss due to Hard Constraint) ============
def compute_loss(model, x_col, t_col, x_ic, t_ic, u_ic, nu=0.01):
    # 1. PDE residual loss
    f = compute_pde_residual(model, x_col, t_col, nu)
    loss_pde = torch.mean(f ** 2)

    # 2. Initial condition loss
    u_ic_pred = model(x_ic, t_ic)
    loss_ic = torch.mean((u_ic_pred - u_ic) ** 2)

    # 3. Total loss (No BC loss needed)
    # Weight increased for PDE as requested
    loss = 5.0 * loss_pde + 10.0 * loss_ic
    return loss, loss_pde, loss_ic


# ============ 5. Data Generation ============
def generate_data():
    # Collocation points (dynamic resampling)
    x_col = torch.FloatTensor(N_COL, 1).uniform_(*DOMAIN_X)
    t_col = torch.FloatTensor(N_COL, 1).uniform_(*DOMAIN_T)

    # Initial condition: t=0, u=-sin(pi*x)
    x_ic = torch.FloatTensor(N_IC, 1).uniform_(*DOMAIN_X)
    t_ic = torch.zeros(N_IC, 1)
    u_ic = -torch.sin(np.pi * x_ic)

    return x_col, t_col, x_ic, t_ic, u_ic


# ============ 6. Two-Stage Training (Adam + L-BFGS) ============
def train_pinn(model, adam_epochs=10000, lbfgs_epochs=2000):
    print("Starting training: Adam phase...")
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_adam, step_size=5000, gamma=0.5)

    losses = []

    # --- Phase 1: Adam with Dynamic Resampling ---
    for epoch in range(adam_epochs):
        model.train()
        optimizer_adam.zero_grad()

        # Dynamic resampling inside the training loop
        x_col, t_col, x_ic, t_ic, u_ic = generate_data()

        loss, l_pde, l_ic = compute_loss(model, x_col, t_col, x_ic, t_ic, u_ic, nu=NU)

        loss.backward()
        optimizer_adam.step()
        scheduler.step()
        losses.append(loss.item())

        if epoch % 1000 == 0:
            print(f"Adam Epoch {epoch:5d} | Total: {loss.item():.6f} | PDE: {l_pde.item():.6f} | IC: {l_ic.item():.6f}")

    # --- Phase 2: L-BFGS Refinement ---
    print("\nStarting training: L-BFGS phase...")
    # For L-BFGS, we fix the collocation points to ensure convergence
    # Dynamic resampling during L-BFGS breaks the quasi-Newton optimization
    x_col, t_col, x_ic, t_ic, u_ic = generate_data()
    # Transfer to L-BFGS requires the whole dataset to be evaluated
    x_col_lbfgs = x_col.clone()
    t_col_lbfgs = t_col.clone()
    x_ic_lbfgs = x_ic.clone()
    t_ic_lbfgs = t_ic.clone()
    u_ic_lbfgs = u_ic.clone()

    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=1.0,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe"
    )

    for epoch in range(lbfgs_epochs):
        def closure():
            optimizer_lbfgs.zero_grad()
            loss, _, _ = compute_loss(model, x_col_lbfgs, t_col_lbfgs, x_ic_lbfgs, t_ic_lbfgs, u_ic_lbfgs, nu=NU)
            loss.backward()
            return loss

        loss_val = optimizer_lbfgs.step(closure)
        losses.append(loss_val.item())

        if epoch % 200 == 0:
            print(f"L-BFGS Epoch {epoch:5d} | Total Loss: {loss_val.item():.6f}")

    return losses


# ============ 7. Evaluation & Visualization (All English) ============
def evaluate_and_plot(model, losses):
    print("\nEvaluating model...")
    model.eval()

    # 1. Generate test grid
    x_test = torch.linspace(-1, 1, 256).view(-1, 1)
    t_test = torch.linspace(0, 1, 100).view(-1, 1)
    X, T = torch.meshgrid(x_test.squeeze(), t_test.squeeze(), indexing='ij')
    x_flat = X.reshape(-1, 1)
    t_flat = T.reshape(-1, 1)

    # 2. Model Inference
    with torch.no_grad():
        u_pred = model(x_flat, t_flat).reshape(X.shape).numpy()

    X_np = X.numpy()
    T_np = T.numpy()

    # 3. Plot Loss Curve
    plt.figure(figsize=(10, 4))
    plt.semilogy(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log scale)")
    plt.title("PINNs Training Loss - 1D Burgers Equation (Adam + L-BFGS)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_loss_optimized.png", dpi=150)

    # 4. Plot Spatio-temporal Heatmap
    plt.figure(figsize=(10, 5))
    c = plt.contourf(X_np, T_np, u_pred, levels=100, cmap='jet')
    plt.colorbar(c, label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title('PINN Prediction u(x,t) Heatmap')
    plt.tight_layout()
    plt.savefig("predicted_heatmap_optimized.png", dpi=150)

    # 5. Plot Slices at t=0.5 and t=0.8
    idx_05 = int(0.5 * (len(t_test) - 1))
    idx_08 = int(0.8 * (len(t_test) - 1))

    plt.figure(figsize=(10, 5))
    plt.plot(x_test.numpy(), u_pred[:, idx_05], label='t = 0.5 (PINN Prediction)', linewidth=2, color='blue')
    plt.plot(x_test.numpy(), u_pred[:, idx_08], label='t = 0.8 (PINN Prediction)', linewidth=2, color='red')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('PINN Prediction at Specific Time Slices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("predicted_slices_optimized.png", dpi=150)

    print("Evaluation complete! Images saved.")


# ============ 8. Main Execution ============
if __name__ == "__main__":
    # Initialize model
    model = PINN(layers=[2, 128, 128, 128, 1])

    # Train
    losses = train_pinn(model, adam_epochs=10000, lbfgs_epochs=2000)

    # Evaluate and plot
    evaluate_and_plot(model, losses)

    print("All tasks completed successfully.")