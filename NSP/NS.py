import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 解决 PyCharm 中 matplotlib 中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 网络结构
# ==========================================
class NSPINN(nn.Module):
    def __init__(self, layers=[2, 64, 64, 64, 3]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
        # 自适应斜率
        self.alpha = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, x, y):
        inputs = torch.cat([x, y], dim=1)
        for layer in self.layers[:-1]:
            inputs = layer(inputs)
            inputs = torch.tanh(self.alpha * inputs)
        out = self.layers[-1](inputs)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p


# ==========================================
# 2. 物理残差计算 (Navier-Stokes)
# ==========================================
def compute_ns_residuals(model, x, y, Re=100.0):
    x.requires_grad_(True)
    y.requires_grad_(True)

    u, v, p = model(x, y)

    # 一阶导数
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, torch.ones_like(v), create_graph=True)[0]
    p_x = torch.autograd.grad(p, x, torch.ones_like(p), create_graph=True)[0]
    p_y = torch.autograd.grad(p, y, torch.ones_like(p), create_graph=True)[0]

    # 二阶导数
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    # 残差
    f_mx = u * u_x + v * u_y + p_x - (1.0 / Re) * (u_xx + u_yy)
    f_my = u * v_x + v * v_y + p_y - (1.0 / Re) * (v_xx + v_yy)
    f_c = u_x + v_y

    return f_mx, f_my, f_c


# ==========================================
# 3. 边界条件损失 (Lid-driven cavity)
# ==========================================
def compute_bc_loss(model, bc_data):
    """
    针对 Lid-driven cavity 的边界条件实现。
    返回逐样本的残差平方和（拼接后的张量）。
    """
    loss_list = []

    # 1. 顶盖边界: u = 1, v = 0
    if 'top' in bc_data:
        x_top, y_top = bc_data['top']
        u_top, v_top, _ = model(x_top, y_top)
        loss_top = (u_top - 1.0) ** 2 + (v_top - 0.0) ** 2
        loss_list.append(loss_top)

    # 2. 其他三个壁面: u = 0, v = 0
    for wall_key in ['bottom', 'left', 'right']:
        if wall_key in bc_data:
            x_w, y_w = bc_data[wall_key]
            u_w, v_w, _ = model(x_w, y_w)
            loss_w = (u_w - 0.0) ** 2 + (v_w - 0.0) ** 2
            loss_list.append(loss_w)

    # 在维度0上进行拼接，确保 shape 一致
    loss_bc = torch.cat(loss_list, dim=0)
    return loss_bc


# ==========================================
# 4. 复合损失函数
# ==========================================
def compute_ns_loss(model, x_col, y_col, bc_data, Re=100.0):
    f_mx, f_my, f_c = compute_ns_residuals(model, x_col, y_col, Re)

    # 逐样本的 PDE 残差平方和
    loss_pde = f_mx ** 2 + f_my ** 2 + f_c ** 2  # (N, 1)

    # 逐样本的 BC 残差
    loss_bc = compute_bc_loss(model, bc_data)  # (N_bc, 1)

    return loss_pde, loss_bc


# ==========================================
# 5. GradNorm 自适应权重
# ==========================================
class GradNormBalancer:
    def __init__(self, num_tasks=2, alpha=1.5):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.initial_losses = None

    def compute_grad_norms(self, loss, shared_layer):
        params = list(shared_layer.parameters())
        grads = torch.autograd.grad(
            loss, params,
            retain_graph=True, allow_unused=True
        )
        norm = 0.0
        for g in grads:
            if g is not None:
                norm += g.norm() ** 2
        return norm ** 0.5

    def update_weights(self, loss_values, shared_layer):
        loss_tensor = torch.stack(loss_values)

        if self.initial_losses is None:
            self.initial_losses = loss_tensor.detach().clone()

        grad_norms = []
        for loss in loss_values:
            gn = self.compute_grad_norms(loss, shared_layer)
            grad_norms.append(gn)
        grad_norms = torch.stack(grad_norms)

        loss_ratios = loss_tensor.detach() / (self.initial_losses + 1e-8)
        mean_ratio = loss_ratios.mean()

        target_norms = (loss_ratios / mean_ratio) ** self.alpha

        self.weights.data = (
                self.weights.data * (target_norms / (grad_norms.detach() + 1e-8))
        )
        self.weights.data = (
                self.weights.data / (self.weights.data.sum() + 1e-8) * self.num_tasks
        )
        return self.weights


# ==========================================
# 6. RBA 自适应权重
# ==========================================
class RBABalancer:
    def __init__(self, eta=0.001, gamma=0.999):
        self.eta = eta
        self.gamma = gamma
        self.lambdas = {}

    def update(self, key, residuals):
        residual_magnitude = residuals.detach().abs()

        if key not in self.lambdas:
            self.lambdas[key] = torch.zeros_like(residual_magnitude)

        self.lambdas[key] = (
                self.gamma * self.lambdas[key]
                + self.eta * residual_magnitude
        )

        # 使用均值归一化，避免 Batch 间剧烈震荡
        mean_val = self.lambdas[key].mean()
        self.lambdas[key] = self.lambdas[key] / (mean_val + 1e-8)

        return self.lambdas[key]


# ==========================================
# 7. 训练步骤
# ==========================================
def train_step(model, optimizer, x_col, y_col, bc_data, Re,
               balancer_type='rba', gradnorm_balancer=None, rba_balancer=None):
    optimizer.zero_grad()

    # 1. 计算逐样本损失
    loss_pde_samples, loss_bc_samples = compute_ns_loss(model, x_col, y_col, bc_data, Re)

    # 2. 聚合为标量损失
    loss_pde = loss_pde_samples.mean()
    loss_bc = loss_bc_samples.mean()

    # 3. 应用权重策略
    if balancer_type == 'fixed':
        lambda_pde, lambda_bc = 1.0, 100.0
        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc

    elif balancer_type == 'gradnorm':
        w = gradnorm_balancer.update_weights([loss_pde, loss_bc], model.layers[0])
        total_loss = w[0] * loss_pde + w[1] * loss_bc

    elif balancer_type == 'rba':
        w_pde = rba_balancer.update('pde', loss_pde_samples)
        w_bc = rba_balancer.update('bc', loss_bc_samples)
        total_loss = (w_pde * loss_pde_samples).mean() + (w_bc * loss_bc_samples).mean()

    else:
        raise ValueError("Unknown balancer type")

    # 4. 反向传播与优化
    total_loss.backward()
    optimizer.step()

    return total_loss.item(), loss_pde.item(), loss_bc.item()


# ==========================================
# 8. 数据生成与可视化
# ==========================================
def generate_data(device, N_col=2000, N_bc=200):
    """生成 Lid-driven cavity 的网格与边界数据"""
    # 内部配点 (均匀网格 + 随机扰动以增加泛化)
    x = torch.rand(N_col, 1, device=device)
    y = torch.rand(N_col, 1, device=device)

    # 顶盖 (y=1, u=1, v=0)
    x_top = torch.linspace(0, 1, N_bc, device=device).reshape(-1, 1)
    y_top = torch.ones(N_bc, 1, device=device)

    # 底部 (y=0, u=0, v=0)
    x_bottom = torch.linspace(0, 1, N_bc, device=device).reshape(-1, 1)
    y_bottom = torch.zeros(N_bc, 1, device=device)

    # 左壁 (x=0, u=0, v=0)
    x_left = torch.zeros(N_bc, 1, device=device)
    y_left = torch.linspace(0, 1, N_bc, device=device).reshape(-1, 1)

    # 右壁 (x=1, u=0, v=0)
    x_right = torch.ones(N_bc, 1, device=device)
    y_right = torch.linspace(0, 1, N_bc, device=device).reshape(-1, 1)

    bc_data = {
        'top': (x_top, y_top),
        'bottom': (x_bottom, y_bottom),
        'left': (x_left, y_left),
        'right': (x_right, y_right)
    }

    return (x, y), bc_data


def plot_results(model, device, Re=100):
    """绘制预测的流场云图、流线图和涡量图"""
    model.eval()
    # 生成高分辨率网格用于绘图
    x = torch.linspace(0, 1, 200)
    y = torch.linspace(0, 1, 200)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    x_flat = X.reshape(-1, 1).to(device)
    y_flat = Y.reshape(-1, 1).to(device)

    with torch.no_grad():
        u, v, p = model(x_flat, y_flat)

    u = u.cpu().numpy().reshape(200, 200)
    v = v.cpu().numpy().reshape(200, 200)
    p = p.cpu().numpy().reshape(200, 200)

    # 提取 1D 坐标数组 (供 streamplot 和 contourf 使用)
    x_np = x.numpy()
    y_np = y.numpy()

    # ================= 绘图 1: 云图 =================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    cf1 = axes[0].contourf(x_np, y_np, u.T, levels=50, cmap='jet')
    axes[0].set_title(f'u 速度场 (Re={Re})')
    fig.colorbar(cf1, ax=axes[0])

    cf2 = axes[1].contourf(x_np, y_np, v.T, levels=50, cmap='jet')
    axes[1].set_title(f'v 速度场 (Re={Re})')
    fig.colorbar(cf2, ax=axes[1])

    cf3 = axes[2].contourf(x_np, y_np, p.T, levels=50, cmap='jet')
    axes[2].set_title(f'p 压力场 (Re={Re})')
    fig.colorbar(cf3, ax=axes[2])

    plt.tight_layout()

    # ================= 绘图 2: 流线图 =================
    plt.figure(figsize=(7, 6))
    plt.streamplot(x_np, y_np, u.T, v.T, density=2.0, color='k', linewidth=1, arrowsize=1.5)
    plt.contourf(x_np, y_np, u.T, levels=50, cmap='jet', alpha=0.6)
    plt.colorbar(label='u velocity')
    plt.title(f'流线图 (Streamlines, Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')

    # ================= 绘图 3: 涡量图 (修复版) =================
    # 注意：计算涡量必须重新开启梯度追踪，不能在 no_grad 下进行
    x_v = X.reshape(-1, 1).to(device).requires_grad_(True)
    y_v = Y.reshape(-1, 1).to(device).requires_grad_(True)

    u_v, v_v, _ = model(x_v, y_v)

    # 修复：一次性同时计算两个导数，避免计算图被提前释放
    # 这里 inputs 的顺序是 [y_v, x_v]，对应 outputs [u_v, v_v]
    grads = torch.autograd.grad(
        outputs=[u_v, v_v],
        inputs=[y_v, x_v],
        grad_outputs=[torch.ones_like(u_v), torch.ones_like(v_v)],
        create_graph=False
    )
    u_y = grads[0]
    v_x = grads[1]

    # 涡量 omega = dv/dx - du/dy
    omega = (v_x - u_y).cpu().numpy().reshape(200, 200)

    plt.figure(figsize=(7, 6))
    # 涡量有正有负，使用 'seismic' 双色图谱更直观
    # 同样需要转置 omega.T 并使用一维坐标数组
    cf_omega = plt.contourf(x_np, y_np, omega.T, levels=50, cmap='seismic')
    plt.colorbar(cf_omega, label='Vorticity')
    plt.title(f'涡量场 (Vorticity, Re={Re})')
    plt.xlabel('x')
    plt.ylabel('y')

    # 显示所有图形
    plt.show()


# ==========================================
# 9. 主程序入口
# ==========================================
if __name__ == '__main__':
    # 设置随机种子，保证结果可复现
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 实例化网络模型
    model = NSPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    # 生成数据
    (x_col, y_col), bc_data = generate_data(device, N_col=3000, N_bc=200)

    # 实例化权重调节器 (选择 'fixed', 'gradnorm', 或 'rba')
    BALANCER_TYPE = 'rba'  # 强烈建议使用 'rba' 或 'gradnorm'
    rba_balancer = RBABalancer(eta=0.001, gamma=0.999)
    gradnorm_balancer = GradNormBalancer(num_tasks=2, alpha=1.5)

    # 训练循环
    epochs = 15000  # 实际训练需要较长时间，可以调至 20000 - 50000
    print(f"开始训练，使用策略: {BALANCER_TYPE} ...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, loss_pde, loss_bc = train_step(
            model=model,
            optimizer=optimizer,
            x_col=x_col,
            y_col=y_col,
            bc_data=bc_data,
            Re=100.0,
            balancer_type=BALANCER_TYPE,
            rba_balancer=rba_balancer,
            gradnorm_balancer=gradnorm_balancer
        )

        scheduler.step()

        # 每 1000 个 epoch 打印一次训练状态
        if epoch % 1000 == 0:
            print(f"Epoch [{epoch:5d}/{epochs}] | "
                  f"Total Loss: {total_loss:.6f} | "
                  f"PDE Loss: {loss_pde:.6f} | "
                  f"BC Loss: {loss_bc:.6f}")

    print("训练完成！开始绘制结果...")
    plot_results(model, device, Re=100)