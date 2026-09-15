import pandas as pd
import matplotlib.pyplot as plt

# 1. 设置全局字体，防止中文字符在图表标题中显示为方块
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows 系统自带的黑体
plt.rcParams['axes.unicode_minus'] = False    # 正常显示坐标轴上的负号

# 2. 读取本地数据文件 (路径前加 r 防止反斜杠转义)
file_path = "240410-240412数据.xls"
data = pd.read_excel(file_path)

# 3. 提取变量数据
# 使用 iloc[:, 索引号] 来提取列，彻底避开 Excel 表头符号乱码的问题
# 注意：Python 的 iloc 索引是从 0 开始的
# A列=0, B列=1, C列=2, D列=3... 依此类推
Time = data.iloc[:, 2]  # C列：时间
Ra = data.iloc[:, 3]    # D列：总辐射
wd = data.iloc[:, 4]    # E列：风向(°)
ws = data.iloc[:, 5]    # F列：风速(m/s)
Rf = data.iloc[:, 6]    # G列：雨量(mm)
T = data.iloc[:, 7]     # H列：温度(℃)
RH = data.iloc[:, 8]    # I列：湿度(%RH)
e = data.iloc[:, 9]     # J列：蒸发量(mm)

# 定义辅助函数，完全模拟 MATLAB 的 movmean (居中滑动平均)
def movmean(series, window):
    return series.rolling(window=window, center=True, min_periods=1).mean()

# ====== Figure 1: 辐射量与温度 ======
plt.figure(1, figsize=(10, 8))

# 子图1：辐射量 (过滤掉大于等于300的异常值)
plt.subplot(2, 1, 1)
mask = Ra < 300
plt.plot(Time[mask], Ra[mask], label='原始数据')
plt.plot(Time[mask], movmean(Ra[mask], 30), linewidth=1.5, label='30点滑动平均')
plt.title('辐射量')
plt.ylabel('W/m2')
plt.xlabel('时间')
plt.legend()

# 子图2：温度
plt.subplot(2, 1, 2)
plt.plot(Time, T, label='原始数据')
plt.plot(Time, movmean(T, 30), linewidth=1.5, label='30点滑动平均')
plt.title('温度')
plt.ylabel('°C')
plt.xlabel('时间')
plt.legend()
plt.tight_layout()

# ====== Figure 2: 风速与风向 ======
plt.figure(2, figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(Time, ws, label='原始数据')
plt.plot(Time, movmean(ws, 20), linewidth=1.5, label='20点滑动平均')
plt.title('风速')
plt.ylabel('m/s')
plt.xlabel('时间')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Time, wd, label='原始数据')
plt.plot(Time, movmean(wd, 20), linewidth=1.5, label='20点滑动平均')
plt.title('风向')
plt.ylabel('°')
plt.xlabel('时间')
plt.legend()
plt.tight_layout()

# ====== Figure 3: 相对湿度与降水量 ======
plt.figure(3, figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(Time, RH, label='原始数据')
plt.plot(Time, movmean(RH, 10), linewidth=1.5, label='10点滑动平均')
plt.title('相对湿度')
plt.ylabel('%')
plt.xlabel('时间')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(Time, Rf, label='原始数据')
plt.plot(Time, movmean(Rf, 10), linewidth=1.5, label='10点滑动平均')
plt.title('降水量')
plt.ylabel('mm')
plt.xlabel('时间')
plt.legend()
plt.tight_layout()

# ====== Figure 4: 蒸发量 ======
plt.figure(4, figsize=(10, 4))
plt.plot(Time, e, label='原始数据')
plt.plot(Time, movmean(e, 10), linewidth=1.5, label='10点滑动平均')
plt.title('蒸发量')
plt.ylabel('mm')
plt.xlabel('时间')
plt.legend()
plt.tight_layout()

# 4. 触发渲染，弹出所有图表窗口
plt.show()