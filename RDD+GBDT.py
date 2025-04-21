import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def load_and_enhance_data():
    # 保持原有数据增强逻辑不变
    df = pd.read_excel("/Users/anran/Desktop/14年4季度-23年2季度/14-23log.xlsx", na_values=["Does not apply", "No Answer", "Not applicable", "No answer"])
    df['date'] = pd.to_datetime(df['date'].astype(str).str.replace(r'(\d+)\.(\d+)', r'\1-\2-01'), errors='coerce')

    df = df[(df['date'] >= '2014-12-01') & df['WorkType'].isin(['Full-time', 'Part-time']) & df['HourlyPay'].notna()]

    # 时间特征工程
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['HOURPAY_lag1'] = df.groupby('WorkType')['HourlyPay'].transform(lambda x: x.shift(1).bfill())
    df['HOURPAY_ma3'] = df.groupby('WorkType')['HourlyPay'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    # 政策变量增强
    policy_date = pd.to_datetime('2020-04-01')
    df['post_policy'] = (df['date'] >= policy_date).astype(int)
    df['months_since_policy'] = (df['date'] - policy_date).dt.days / 30
    df['log_months_since_policy'] = np.log(np.abs(df['months_since_policy']) + 1) * np.sign(df['months_since_policy'])
    df['policy_effect'] = df['post_policy'] * df['log_months_since_policy']

    return df


def build_preprocessor():
    # 保持原有预处理逻辑不变
    categorical_features = [
        'Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector',
        'EduLevel', 'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome'
    ]
    numerical_features = [
        'year', 'month', 'quarter', 'month_sin', 'month_cos',
        'HOURPAY_lag1', 'HOURPAY_ma3', 'log_months_since_policy',
        'post_policy', 'policy_effect'
    ]

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(
            handle_unknown='infrequent_if_exist',
            sparse_output=False,
            drop='if_binary'
        ), categorical_features)
    ], remainder='drop')

    return preprocessor, categorical_features, numerical_features


def calculate_policy_effect(model, preprocessor, X_sample, categorical_features):
    """保持原有政策效应计算逻辑"""
    policy_scenarios = pd.DataFrame({
        'log_months_since_policy': np.linspace(-3, 3, 100),
        'post_policy': [0] * 50 + [1] * 50,
        'policy_effect': [0] * 50 + list(np.linspace(0, 3, 50))
    })

    # 填充其他特征
    for col in X_sample.columns:
        if col not in policy_scenarios:
            if col in categorical_features:
                policy_scenarios[col] = X_sample[col].mode()[0]
            else:
                policy_scenarios[col] = X_sample[col].median()

    # 确保分类变量为字符串
    for col in categorical_features:
        if col in policy_scenarios:
            policy_scenarios[col] = policy_scenarios[col].astype(str)

    X_transformed = preprocessor.transform(policy_scenarios)
    predictions = model.predict(X_transformed)

    # 计算政策效应
    pre_policy_mean = predictions[:50].mean()
    post_policy_mean = predictions[50:].mean()
    policy_effect = post_policy_mean - pre_policy_mean

    # 使用SHAP计算标准差
    explainer = shap.Explainer(model)
    shap_values = explainer(X_transformed)

    # 获取政策相关特征的SHAP值
    policy_feature_indices = [
        list(preprocessor.get_feature_names_out()).index(f)
        for f in ['post_policy', 'policy_effect']
        if f in preprocessor.get_feature_names_out()
    ]
    policy_shap = shap_values.values[:, policy_feature_indices].sum(axis=1)

    return {
        'effect_size': policy_effect,
        'std_dev': policy_shap[50:].std(),
        'pre_policy_mean': pre_policy_mean,
        'post_policy_mean': post_policy_mean,
        'scenarios': policy_scenarios,
        'predictions': predictions
    }


def run_gbdt_analysis(df):
    results = {}

    for work_type in ["Full-time", "Part-time"]:
        print(f"\n🚀 训练GBDT模型: {work_type}")
        sub_df = df[df["WorkType"] == work_type].copy()

        feature_columns = [
            'Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector', 'EduLevel',
            'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome',
            'year', 'month', 'quarter', 'month_sin', 'month_cos',
            'HOURPAY_lag1', 'HOURPAY_ma3', 'log_months_since_policy',
            'post_policy', 'policy_effect'
        ]

        X = sub_df[feature_columns]
        y = sub_df['HourlyPay']

        # 分类变量处理
        categorical_cols = ['Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector',
                            'EduLevel', 'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome']
        for col in categorical_cols:
            X[col] = X[col].astype(str)

        preprocessor, categorical_features, numerical_features = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

        # 关键修改点：使用GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            min_samples_split=10,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 评估指标计算
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        epsilon = 1e-10
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + epsilon))) * 100

        print("\n📊 模型评估指标:")
        print(f"MSE: {mse:.4f}  RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}  MAPE: {mape:.2f}%")
        print(f"R²: {r2:.4f}")

        # 政策效应分析
        policy_result = calculate_policy_effect(model, preprocessor, X, categorical_features)

        # 存储结果
        results[work_type] = {
            **policy_result,
            'metrics': {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape,
                'R2': r2
            }
        }

        print(f"\n📌 {work_type} 政策效应分析:")
        print(f"政策前平均小时工资: {policy_result['pre_policy_mean']:.2f}")
        print(f"政策后平均小时工资: {policy_result['post_policy_mean']:.2f}")
        print(f"政策效应大小: {policy_result['effect_size']:.2f} ± {policy_result['std_dev']:.2f}")
        print(f"效应百分比变化: {(policy_result['effect_size'] / policy_result['pre_policy_mean']) * 100:.1f}%")

        # 可视化部分保持不变
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=sub_df['log_months_since_policy'],
                        y=sub_df['HourlyPay'],
                        hue=sub_df['post_policy'],
                        palette={0: 'blue', 1: 'red'},
                        alpha=0.3)
        plt.plot(policy_result['scenarios']['log_months_since_policy'],
                 policy_result['predictions'],
                 color='black', linewidth=2, label='Model Prediction')
        pre_mask = policy_result['scenarios']['post_policy'] == 0
        post_mask = policy_result['scenarios']['post_policy'] == 1
        plt.plot(policy_result['scenarios'][pre_mask]['log_months_since_policy'],
                 policy_result['predictions'][pre_mask],
                 color='blue', linewidth=2, linestyle='--', label='Pre-policy Trend')
        plt.plot(policy_result['scenarios'][post_mask]['log_months_since_policy'],
                 policy_result['predictions'][post_mask],
                 color='red', linewidth=2, linestyle='--', label='Post-policy Trend')
        plt.axvline(0, color='green', linestyle='--', label='Policy Implementation')
        plt.xlabel("Months Since Policy", fontsize=12)
        plt.ylabel(r'$\log(\text{Hourly Pay})$', fontsize=12)
        plt.legend()
        save_path = f"/Users/anran/Desktop/4.1/RDD/RDD_GBDT_{'Fulltime' if work_type == 'Full-time' else 'Parttime'}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    return results


def main():
    df = load_and_enhance_data()
    results = run_gbdt_analysis(df)

    print("\n🔍 最终政策效应总结:")
    for work_type, res in results.items():
        print(f"\n{work_type}:")
        print(f"• 效应值: {res['effect_size']:.4f} ± {res['std_dev']:.4f}")
        print(f"• 变化率: {(res['effect_size'] / res['pre_policy_mean']) * 100:.2f}%")
        print("• 模型指标:")
        print(f"  MSE: {res['metrics']['MSE']:.4f}  RMSE: {res['metrics']['RMSE']:.4f}")
        print(f"  MAE: {res['metrics']['MAE']:.4f}  MAPE: {res['metrics']['MAPE']:.2f}%")
        print(f"  R²: {res['metrics']['R2']:.4f}")


if __name__ == "__main__":
    main()
