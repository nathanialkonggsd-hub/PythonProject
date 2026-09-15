# plot_all.py
# 运行: python plot_all.py
# 生成: soft_constraint_loss.png, cavity_setup.png,
#       total_loss_comparison.png, training_loss_comparison.png,
#       ntk_spectrum.png, pred_vs_ref.png

import numpy as np
import matplotlib.pyplot as plt

# 全局设置
plt.rcParams['font.size'] = 11
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.autolayout'] = True

# ============================================================
# 图1: soft_constraint_loss.png
# 对应文章 1.3 节：软约束分项损失与总损失对比
# ============================================================
def plot_soft_constraint_loss():
    epochs = np.arange(0, 15000, 100)
    # 模拟软约束训练中的 loss 演化
    loss_pde = 5e-2 * np.exp(-epochs / 3000) + 1e-3
    loss_bc  = 8e-2 * np.exp(-epochs / 8000) + 5e-3
    loss_total = loss_pde + 100 * loss_bc   # lambda_bc = 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 分项损失
    axes[0].semilogy(epochs, loss_pde, 'b-', label=r'$L_{PDE}$', linewidth=2)
    axes[0].semilogy(epochs, loss_bc,  'r-', label=r'$L_{BC}$',  linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (log scale)')
    axes[0].set_title('Soft Constraint: Component Losses')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 右图: 总损失对比硬约束
    axes[1].semilogy(epochs, loss_total, 'k-', label='Total (Soft)', linewidth=2)
    axes[1].axhline(y=3.77e-6, color='g', linestyle='--',
                    label='Hard BC Total Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Total Loss: Soft vs Hard')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.savefig('soft_constraint_loss.png', bbox_inches='tight')
    plt.close()
    print('已生成: soft_constraint_loss.png')


# ============================================================
# 图2: cavity_setup.png
# 对应文章 3.1 节：lid-driven cavity 问题设定示意图
# ============================================================
def plot_cavity_setup():
    fig, ax = plt.subplots(figsize=(6, 6))

    # 域
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False,
                               edgecolor='black', linewidth=2))

    # 边界条件标注
    ax.annotate('u=1, v=0', xy=(0.5, 1.02), ha='center',
                fontsize=12, color='red', fontweight='bold')
    ax.annotate('u=0, v=0', xy=(0.5, -0.06), ha='center',
                fontsize=11, color='blue')
    ax.annotate('u=0\nv=0', xy=(-0.08, 0.5), ha='center',
                fontsize=11, color='blue')
    ax.annotate('u=0\nv=0', xy=(1.08, 0.5), ha='center',
                fontsize=11, color='blue')

    # 配点示意
    np.random.seed(42)
    x_pts = np.random.rand(200)
    y_pts = np.random.rand(200)
    ax.scatter(x_pts, y_pts, s=1, alpha=0.3, color='gray')

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.1, 1.12)
    ax.set_aspect('equal')
    ax.set_title('Lid-Driven Cavity: Problem Setup', fontsize=14)
    plt.savefig('cavity_setup.png', bbox_inches='tight')
    plt.close()
    print('已生成: cavity_setup.png')


# ============================================================
# 图3: total_loss_comparison.png
# 对应文章 3.2 节：软约束 vs 硬约束总损失柱状对比
# ============================================================
def plot_total_loss_comparison():
    methods = ['Soft-BC\n(λ=100)', 'Hard-BC']
    total_loss = [1.1e-2, 3.77e-6]
    colors = ['#e74c3c', '#27ae60']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(methods, total_loss, color=colors, width=0.5)
    ax.set_yscale('log')
    ax.set_ylabel('Total Loss (log scale)', fontsize=13)
    ax.set_title('Total Loss: Soft-BC vs Hard-BC (Re=100)', fontsize=14, pad=20)

    for bar, val in zip(bars, total_loss):
        ax.text(bar.get_x() + bar.get_width()/2, val * 1.0,
                f'{val:.2e}', ha='center', fontsize=12, fontweight='bold')

    ax.grid(True, alpha=0.3, axis='y')
    plt.savefig('total_loss_comparison.png', bbox_inches='tight')
    plt.close()
    print('已生成: total_loss_comparison.png')


# ============================================================
# 图4: training_loss_comparison.png
# 对应文章 3.3 节：软/硬约束训练损失曲线对比
# ============================================================
def plot_training_loss_comparison():
    epochs = np.arange(0, 30000, 100)

    # 软约束总损失
    loss_pde_soft = 1e-2 * np.exp(-epochs / 5000) + 2e-3
    loss_bc_soft  = 5e-2 * np.exp(-epochs / 10000) + 8e-3
    loss_total_soft = loss_pde_soft + 100 * loss_bc_soft

    # 硬约束总损失 ≈ 仅 PDE 残差
    loss_pde_hard = 5e-3 * np.exp(-epochs / 2000) + 3e-6

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogy(epochs, loss_total_soft, 'r-', label='Soft-BC Total', linewidth=2)
    ax.semilogy(epochs, loss_pde_hard, 'g-', label='Hard-BC Total (= PDE only)', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss (log scale)', fontsize=12)
    ax.set_title('Training Loss: Soft vs Hard', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.savefig('training_loss_comparison.png', bbox_inches='tight')
    plt.close()
    print('已生成: training_loss_comparison.png')


# ============================================================
# 图5: ntk_spectrum.png
# 对应文章 4.5 节：NTK 特征谱对比
# ============================================================
def plot_ntk_spectrum():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    n_eigs = 100

    # 好的 B 函数：特征值缓慢衰减
    eig_good = np.exp(-np.arange(n_eigs) / 40) * 1e-3
    r_eff_good = (eig_good.sum()**2) / (eig_good**2).sum()

    # 差的 B 函数：谱塌缩
    eig_bad = np.exp(-np.arange(n_eigs) / 5) * 1e-3
    r_eff_bad = (eig_bad.sum()**2) / (eig_bad**2).sum()

    axes[0].semilogy(eig_good, 'b-', linewidth=2)
    axes[0].set_xlabel('Eigenvalue Index', fontsize=12)
    axes[0].set_ylabel('Eigenvalue (log scale)', fontsize=12)
    axes[0].set_title(f'Good B: $r_{{eff}}$ = {r_eff_good:.1f}', fontsize=13)
    axes[0].grid(True, alpha=0.3)
    axes[0].fill_between(range(n_eigs), eig_good, alpha=0.2, color='blue')

    axes[1].semilogy(eig_bad, 'r-', linewidth=2)
    axes[1].set_xlabel('Eigenvalue Index', fontsize=12)
    axes[1].set_ylabel('Eigenvalue (log scale)', fontsize=12)
    axes[1].set_title(f'Bad B (Spectral Collapse): $r_{{eff}}$ = {r_eff_bad:.1f}',
                      fontsize=13)
    axes[1].grid(True, alpha=0.3)
    axes[1].fill_between(range(n_eigs), eig_bad, alpha=0.2, color='red')

    plt.savefig('ntk_spectrum.png', bbox_inches='tight')
    plt.close()
    print('已生成: ntk_spectrum.png')


# ============================================================
# 图6: pred_vs_ref.png (额外补充)
# 预测解 vs 参考解示意对比
# 注意：这里用模拟数据展示效果，真实场景需替换为训练结果和参考解
# ============================================================
def plot_pred_vs_ref():
    # 生成网格
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)

    # 模拟预测解：用两个正弦函数叠加模拟 lid-driven cavity 的 u 场
    u_pred = np.sin(np.pi * X) * np.sin(np.pi * Y) + 0.3 * np.sin(2*np.pi*X)
    v_pred = -np.cos(np.pi * X) * np.sin(np.pi * Y) * 0.5

    # 模拟参考解：在预测解上加一点微小扰动
    np.random.seed(0)
    u_ref = u_pred + 0.02 * np.random.randn(*u_pred.shape)
    v_ref = v_pred + 0.02 * np.random.randn(*v_pred.shape)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 预测 u
    cf1 = axes[0, 0].contourf(X, Y, u_pred, levels=50, cmap='jet')
    axes[0, 0].set_title('Predicted u')
    plt.colorbar(cf1, ax=axes[0, 0])

    # 参考 u
    cf2 = axes[0, 1].contourf(X, Y, u_ref, levels=50, cmap='jet')
    axes[0, 1].set_title('Reference u')
    plt.colorbar(cf2, ax=axes[0, 1])

    # 预测 v
    cf3 = axes[1, 0].contourf(X, Y, v_pred, levels=50, cmap='jet')
    axes[1, 0].set_title('Predicted v')
    plt.colorbar(cf3, ax=axes[1, 0])

    # 参考 v
    cf4 = axes[1, 1].contourf(X, Y, v_ref, levels=50, cmap='jet')
    axes[1, 1].set_title('Reference v')
    plt.colorbar(cf4, ax=axes[1, 1])

    plt.savefig('pred_vs_ref.png', bbox_inches='tight')
    plt.close()
    print('已生成: pred_vs_ref.png')


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':
    print('开始生成第三篇文章所需图片...')
    plot_soft_constraint_loss()
    plot_cavity_setup()
    plot_total_loss_comparison()
    plot_training_loss_comparison()
    plot_ntk_spectrum()
    plot_pred_vs_ref()
    print('全部图片生成完成。')