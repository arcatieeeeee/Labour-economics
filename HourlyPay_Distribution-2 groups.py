import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# 加载和预处理数据
def load_and_enhance_data():
    df = pd.read_excel("/Users/anran/Desktop/14年4季度-23年2季度/14-23xin.xlsx",
                       na_values=["Does not apply", "No Answer", "Not applicable", "No answer"])
    print("\n✅ 数据加载成功！")
    print(f"🔹 数据规模: {df.shape}")
    print(f"🔹 列名: {list(df.columns)}")

    df['date'] = df['date'].astype(str).str.replace(r'(\d+)\.(\d+)', r'\1-\2-01', regex=True)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    print(f"\n📆 数据时间范围: {df['date'].min().date()} ➝ {df['date'].max().date()}")

    df = df[df['WorkType'].isin(['Full-time', 'Part-time'])].copy()
    print(f"🔹 过滤无效 WorkType 后，数据规模: {df.shape}")

    df = df[df['HourlyPay'].notna()].copy()
    print(f"🔹 过滤 HourlyPay 缺失值后，数据规模: {df.shape}")

    if df.empty:
        raise ValueError("❌ 过滤后数据为空，请检查数据源！")

    df['post_policy'] = (df['date'] >= '2020-04-01').astype(int)
    return df

# 找相邻月份样本
def get_adjacent_months_data(df, policy_date):
    def find_valid_month(start_date, direction):
        while True:
            candidate = df[df['date'] == start_date]
            if len(candidate) >= 200:
                return candidate
            start_date += pd.DateOffset(months=direction)
            if start_date < df['date'].min() or start_date > df['date'].max():
                print(f"⚠️ 超出数据范围，找不到满足条件的相邻月份 ({start_date.date()})")
                return pd.DataFrame()

    pre = find_valid_month(policy_date - pd.DateOffset(months=1), -1)
    post = find_valid_month(policy_date + pd.DateOffset(months=1), 1)
    return pre, post

# 图①：分组频率分布图
def plot_hourlypay_frequency(df):
    policy_date = pd.to_datetime('2020-04-01')
    pre_data, post_data = get_adjacent_months_data(df, policy_date)

    if pre_data.empty or post_data.empty:
        print("❌ 无法绘制图表：相邻月份样本不足")
        return

    pre_data = pre_data[pre_data['HourlyPay'] <= 100]
    post_data = post_data[post_data['HourlyPay'] <= 100]
    bins = np.linspace(0, 50, 25)
    colors = {0: "#4C72B0", 1: "#E15759"}

    for group, title in zip(["Full-time", "Part-time"], ["Full-time Workers", "Part-time Workers"]):
        fig, ax = plt.subplots(figsize=(8, 6))
        pre = pre_data[pre_data['WorkType'] == group]
        post = post_data[post_data['WorkType'] == group]

        sns.histplot(pre['HourlyPay'], bins=bins, color=colors[0], label="Pre-Policy", alpha=0.6, ax=ax)
        sns.histplot(post['HourlyPay'], bins=bins, color=colors[1], label="Post-Policy", alpha=0.6, ax=ax)

        ax.axvline(pre['HourlyPay'].mean(), color=colors[0], linestyle='--', label="Pre Mean")
        ax.axvline(post['HourlyPay'].mean(), color=colors[1], linestyle='--', label="Post Mean")

        ax.set_title(f"HourlyPay Distribution: {title}")
        ax.set_xlabel("Hourly Pay (£)")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, 50)
        ax.legend()
        plt.tight_layout()
        plt.show()


# 主函数入口
def main():
    df = load_and_enhance_data()

    print("\n📊 数据质量报告:")
    print(f"🔹 总样本数: {len(df)}")
    print(f"🔹 Full-time 样本数: {len(df[df['WorkType'] == 'Full-time'])}")
    print(f"🔹 Part-time 样本数: {len(df[df['WorkType'] == 'Part-time'])}")
    print("\n📌 关键特征统计:\n", df[['HourlyPay', 'post_policy']].describe().round(2))

    print("\n📊 绘制 HourlyPay 分布图（含均值）")
    plot_hourlypay_frequency(df)

    print("\n📊 绘制均值对比柱状图")
    plot_grouped_mean_bar(df)


if __name__ == "__main__":
    main()
