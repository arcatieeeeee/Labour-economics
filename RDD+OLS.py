import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


# 加载和预处理数据
def load_and_enhance_data():
    df = pd.read_excel("/Users/anran/Desktop/14年4季度-23年2季度/14-23log.xlsx", na_values=["Does not apply", "No Answer", "Not applicable", "No answer"])
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
    df['months_since_policy'] = (df['date'] - pd.to_datetime('2020-04-01')).dt.days / 30

    # 取对数变换（+1 避免 log(0)）
    df['log_months_since_policy'] = np.log(np.abs(df['months_since_policy']) + 1) * np.sign(df['months_since_policy'])

    return df


# 计算回归模型的评价指标
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # 乘以100转为百分比
    r2 = r2_score(y_true, y_pred)

    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "MAPE": mape, "R²": r2}


# 进行RDD回归分析
def run_rdd_analysis(df):
    policy_date = pd.to_datetime('2020-04-01')

    for work_type in ["Full-time", "Part-time"]:
        sub_df = df[df["WorkType"] == work_type].copy()

        # 线性不连续回归模型（使用对数变换后的变量）
        model = smf.ols("HourlyPay ~ log_months_since_policy * post_policy", data=sub_df).fit()
        print(f"\n🔍 {work_type} 线性RDD回归结果（对数变换）:\n")
        print(model.summary())

        # 计算预测值
        y_true = sub_df['HourlyPay']
        y_pred = model.predict(sub_df)

        # 计算并打印回归评价指标
        metrics = evaluate_model(y_true, y_pred)
        print(f"\n📊 {work_type} 模型评估指标：")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # 还原 log_months_since_policy
        sub_df['exp_months_since_policy'] = np.sign(sub_df['log_months_since_policy']) * (
                np.exp(np.abs(sub_df['log_months_since_policy'])) - 1)

        # 绘制RDD回归图
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=sub_df['exp_months_since_policy'], y=sub_df['HourlyPay'], alpha=0.3, label="Observed Data")

        # 画拟合线
        x_range = np.linspace(sub_df['exp_months_since_policy'].min(), sub_df['exp_months_since_policy'].max(), 100)
        log_x_range = np.log(np.abs(x_range) + 1) * np.sign(x_range)

        pre_line = model.params['Intercept'] + model.params['log_months_since_policy'] * log_x_range[log_x_range < 0]
        post_line = (model.params['Intercept'] + model.params['post_policy'] +
                     (model.params['log_months_since_policy'] + model.params['log_months_since_policy:post_policy']) *
                     log_x_range[log_x_range >= 0])

        plt.plot(x_range[x_range < 0], pre_line, color='#4C72B0', label="Pre-Policy Fit")
        plt.plot(x_range[x_range >= 0], post_line, color='#E15759', label="Post-Policy Fit")

        plt.axvline(0, color='black', linestyle='--', label="Policy Implementation")
        plt.title(f"RDD Analysis (Log Transform): {work_type}")
        plt.xlabel("Months Since Policy (Recovered)")
        plt.ylabel("Hourly Pay (£)")
        plt.legend()
        plt.show()


# 主函数
def main():
    df = load_and_enhance_data()

    print("\n📊 数据质量报告:")
    print(f"🔹 总样本数: {len(df)}")
    print(f"🔹 Full-time 样本数: {len(df[df['WorkType'] == 'Full-time'])}")
    print(f"🔹 Part-time 样本数: {len(df[df['WorkType'] == 'Part-time'])}")
    print("\n📌 关键特征统计:\n", df[['HourlyPay', 'post_policy', 'months_since_policy', 'log_months_since_policy']].describe().round(2))

    print("\n📊 运行RDD回归分析")
    run_rdd_analysis(df)


if __name__ == "__main__":
    main()
