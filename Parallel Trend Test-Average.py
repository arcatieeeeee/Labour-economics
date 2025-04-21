import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 设置字体
plt.rcParams["font.family"] = "Times New Roman"

# === 1. 加载数据 ===
df = pd.read_excel("/Users/anran/Desktop/14年4季度-23年2季度/Tables/14-23Parallel Test.xlsx")

# === 2. 时间与工资处理 ===
df["quarter"] = pd.PeriodIndex(df["date"], freq="Q")
df["HourlyPay"] = pd.to_numeric(df["HourlyPay"], errors="coerce")
df = df[df["WorkType"].isin(["Full-time", "Part-time"])]
df = df.dropna(subset=["HourlyPay"])
df = df[df["HourlyPay"] > 0]

# === 3. 计算季度平均工资 ===
avg_pay = df.groupby(["quarter", "WorkType"])["HourlyPay"].mean().reset_index()
avg_pay["quarter_dt"] = avg_pay["quarter"].dt.to_timestamp()

# === 4. 绘图开始 ===
fig, ax = plt.subplots(figsize=(12, 6))

colors = {
    "Full-time": "#1f77b4",
    "Part-time": "#d62728"
}

for group in ["Full-time", "Part-time"]:
    subset = avg_pay[avg_pay["WorkType"] == group]
    ax.plot(subset["quarter_dt"], subset["HourlyPay"],
            marker='o', markersize=6, linewidth=2.5,
            label=group, color=colors[group])

# === 5. 添加政策分割线 & 注释 ===
policy_date = pd.Period("2020Q2", freq="Q").to_timestamp()
ax.axvline(x=policy_date, color="gray", linestyle="--", linewidth=1.5, label="Good Work Plan (2020Q2)")

# 添加箭头注释
ax.annotate('Policy Introduced',
            xy=(policy_date, 19),             # 箭头尖端位置
            xytext=(policy_date, 23.5),       # 注释文字位置
            arrowprops=dict(facecolor='gray', arrowstyle='->', linewidth=1.2),
            fontsize=12,
            ha='center')

# === 6. X轴格式设置为年份 ===
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# === 7. 美化 ===
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Average Hourly Pay (£)", fontsize=14)
ax.tick_params(axis='x', rotation=0, labelsize=12)
ax.set_ylim(5, 25)
ax.set_yticks(np.arange(5, 26, 5))
ax.tick_params(axis='both', labelsize=12)

# 图例优化
ax.legend(title="Group", fontsize=12, title_fontsize=13, loc='upper left', frameon=False)

# 网格和边框
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
ax.xaxis.grid(False)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# 保存图像
plt.tight_layout()
plt.savefig("/Users/anran/Desktop/14年4季度-23年2季度/Figure_HourlyPay_Refined_Annotated.jpg", dpi=300, bbox_inches='tight')
plt.show()
