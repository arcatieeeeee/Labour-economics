import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置字体为 Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# === 1. 读取数据 ===
df = pd.read_excel("/Users/anran/Desktop/14年4季度-23年2季度/Tables/14-23Parallel Test.xlsx")

# === 2. 时间处理 ===
df["quarter"] = pd.PeriodIndex(df["date"], freq="Q")

# === 3. 工资处理 ===
df["HourlyPay"] = pd.to_numeric(df["HourlyPay"], errors="coerce")
df = df[df["WorkType"].isin(["Full-time", "Part-time"])]
df = df.dropna(subset=["HourlyPay"])
df = df[df["HourlyPay"] > 0]
df["LogHourlyPay"] = np.log(df["HourlyPay"])

# === 4. 计算平均对数工资 ===
avg_log_pay = df.groupby(["quarter", "WorkType"])["LogHourlyPay"].mean().reset_index()
avg_log_pay["quarter_dt"] = avg_log_pay["quarter"].dt.to_timestamp()

# === 5. 绘图 ===
plt.figure(figsize=(12, 6))

# 使用颜色区分更强的配色
colors = {
    "Full-time": "#1f77b4",     # 深蓝
    "Part-time": "#d62728"      # 鲜红
}

for group in ["Full-time", "Part-time"]:
    subset = avg_log_pay[avg_log_pay["WorkType"] == group]
    plt.plot(subset["quarter_dt"], subset["LogHourlyPay"], marker='o', label=group, color=colors[group], linewidth=2.2)

# === 6. 添加政策时间线 ===
policy_date = pd.Period("2020Q2", freq="Q").to_timestamp()
plt.axvline(x=policy_date, color="gray", linestyle="--", linewidth=1.5, label="Good Work Plan (2020Q2)")

# === 7. 图形美化 ===
plt.xlabel("Year", fontsize=13)
plt.ylabel("Log of Average Hourly Pay (£)", fontsize=13)
plt.xticks(rotation=45, fontsize=11)
plt.yticks(fontsize=11)
plt.ylim(2.0, 3.0)  # 你可以根据实际情况微调
plt.yticks(np.arange(2.0, 3.0, 0.3))  # 每隔0.025设置一个刻度

plt.legend(title="Group", fontsize=11, title_fontsize=12, frameon=True, loc='upper left')

# 去除四周边框线（spines）
for spine in plt.gca().spines.values():
    spine.set_visible(False)

# 10. 保存图像（可选）
plt.savefig("/Users/anran/Desktop/14年4季度-23年2季度/Figure 2"
            ".jpg", dpi=300,  bbox_inches='tight')


# 11. 显示图像
plt.show()
