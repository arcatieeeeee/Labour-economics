import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def load_and_enhance_data():
    df = pd.read_excel("/Users/anran/Desktop/14年4季度-23年2季度/14-23log.xlsx", na_values=["Does not apply", "No Answer", "Not applicable", "No answer"])
    df['date'] = pd.to_datetime(df['date'].astype(str).str.replace(r'(\d+)\.(\d+)', r'\1-\2-01'), errors='coerce')

    df = df[(df['date'] >= '2014-12-01') & df['WorkType'].isin(['Full-time', 'Part-time']) & df['HourlyPay'].notna()]

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['HOURPAY_lag1'] = df.groupby('WorkType')['HourlyPay'].transform(lambda x: x.shift(1).bfill())
    df['HOURPAY_ma3'] = df.groupby('WorkType')['HourlyPay'].transform(lambda x: x.rolling(3, min_periods=1).mean())

    policy_date = pd.to_datetime('2020-04-01')
    df['post_policy'] = (df['date'] >= policy_date).astype(int)
    df['months_since_policy'] = (df['date'] - policy_date).dt.days / 30
    df['log_months_since_policy'] = np.log(np.abs(df['months_since_policy']) + 1) * np.sign(df['months_since_policy'])

    df['policy_effect'] = df['post_policy'] * df['log_months_since_policy']

    return df


def build_preprocessor():
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
    policy_scenarios = pd.DataFrame({
        'log_months_since_policy': np.linspace(-3, 3, 100),
        'post_policy': [0] * 50 + [1] * 50,
        'policy_effect': [0] * 50 + list(np.linspace(0, 3, 50))
    })

    for col in X_sample.columns:
        if col not in policy_scenarios:
            if col in categorical_features:
                policy_scenarios[col] = X_sample[col].mode()[0]
            else:
                policy_scenarios[col] = X_sample[col].median()

    for col in categorical_features:
        if col in policy_scenarios:
            policy_scenarios[col] = policy_scenarios[col].astype(str)

    X_transformed = preprocessor.transform(policy_scenarios)
    predictions = model.predict(X_transformed)

    pre_policy_mean = predictions[:50].mean()
    post_policy_mean = predictions[50:].mean()
    policy_effect = post_policy_mean - pre_policy_mean

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)

    policy_shap = shap_values[:, [
        list(preprocessor.get_feature_names_out()).index(f)
        for f in ['post_policy', 'policy_effect']
        if f in preprocessor.get_feature_names_out()
    ]].sum(axis=1)

    return {
        'effect_size': policy_effect,
        'std_dev': policy_shap[50:].std(),
        'pre_policy_mean': pre_policy_mean,
        'post_policy_mean': post_policy_mean,
        'scenarios': policy_scenarios,
        'predictions': predictions
    }


def run_rf_analysis(df):
    results = {}

    for work_type in ["Full-time", "Part-time"]:
        print(f"\n🚀 训练 随机森林 RDD 模型: {work_type}")
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

        for col in ['Sex', 'Industry', 'MariStatus', 'WorkReg', 'Sector', 'EduLevel', 'WhyPJob', 'Ethnicity', 'Benfts', 'WorkHome']:
            X[col] = X[col].astype(str)

        preprocessor, categorical_features, numerical_features = build_preprocessor()
        X_transformed = preprocessor.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=5, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100

        print("\n📊 评估指标:")
        print(f"✅ MSE: {mse:.4f}")
        print(f"✅ RMSE: {rmse:.4f}")
        print(f"✅ MAE: {mae:.4f}")
        print(f"✅ MAPE: {mape:.2f}%")
        print(f"✅ R²: {r2:.4f}")

        policy_result = calculate_policy_effect(model, preprocessor, X, categorical_features)

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

    return results



def main():
    df = load_and_enhance_data()
    results = run_rf_analysis(df)

    for work_type, res in results.items():
        print(f"\n{work_type}: 政策效应 {res['effect_size']:.4f} ± {res['std_dev']:.4f}")


if __name__ == "__main__":
    main()
