import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# 设置全局字体为Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
# 加载数据
df = pd.read_excel("14-23xin.xlsx",
                   na_values=["Does not apply", "No Answer", "Not applicable", "No answer"]).drop(['date', 'PERSID'],
                                                                                                  axis=1)

# 数据清洗
df['HourlyPay'] = pd.to_numeric(df['HourlyPay'], errors='coerce')
df = df[df['HourlyPay'].notna()]


# 改进的编码（性别单独处理）
def enhanced_encoder(df):
    encoded = df.copy()

    # 对性别进行独热编码
    sex_dummies = pd.get_dummies(encoded['Sex'], prefix='Sex', dummy_na=False)
    encoded = pd.concat([encoded.drop('Sex', axis=1), sex_dummies], axis=1)

    # 处理其他二分类变量
    binary_cols = ['Benfts']
    for col in binary_cols:
        le = LabelEncoder()
        valid_mask = encoded[col].notna()
        try:
            encoded.loc[valid_mask, col] = le.fit_transform(encoded.loc[valid_mask, col].astype(str))
        except:
            encoded[col] = np.nan

    # 处理多分类变量
    cat_cols = ['WorkType', 'Industry', 'MariStatus', 'WorkReg', 'Sector',
                'WhyPJob', 'Ethnicity', 'LessPay', 'WorkHome']

    for col in cat_cols:
        valid_mask = encoded[col].notna()
        dummies = pd.get_dummies(encoded.loc[valid_mask, col].astype(str), prefix=col)
        encoded = pd.concat([encoded.drop(col, axis=1), dummies], axis=1)

    return encoded.dropna(axis=1, how='all')


encoded_df = enhanced_encoder(df)

# 分割数据集
full_time_df = encoded_df[encoded_df['WorkType_Full-time'] == 1].drop(columns=['WorkType_Full-time', 'WorkType_Part-time'])
part_time_df = encoded_df[encoded_df['WorkType_Part-time'] == 1].drop(columns=['WorkType_Full-time', 'WorkType_Part-time'])


# 分析函数（新增保存功能）
def analyze_group(df, group_name):
    results = []

    for col in df.columns:
        if col == 'HourlyPay':
            continue

        clean_data = df[['HourlyPay', col]].dropna()
        if len(clean_data) < 30:
            continue

        try:
            if pd.api.types.is_numeric_dtype(clean_data[col]):
                if clean_data[col].nunique() == 2:
                    corr, pval = stats.pointbiserialr(clean_data['HourlyPay'], clean_data[col])
                else:
                    corr, pval = stats.pearsonr(clean_data['HourlyPay'], clean_data[col])
            else:
                groups = clean_data.groupby(col)['HourlyPay'].apply(list)
                if len(groups) < 2:
                    continue
                f_val, pval = stats.f_oneway(*groups)
                corr = np.sqrt(f_val / (f_val + len(clean_data) - 1))
        except:
            continue

        results.append({
            'Feature': col,
            'Correlation': corr,
            'P-value': pval
        })

    df_results = pd.DataFrame(results)
    significant = df_results[df_results['P-value'] < 0.05].sort_values('Correlation', ascending=False)

    # 可视化及保存设置
    plt.figure(figsize=(12, 8))
    top20 = significant.head(10)
    colors = ['#1f77b4' if x > 0 else '#d62728' for x in top20['Correlation']]

    ax = sns.barplot(x='Correlation', y='Feature', data=top20, palette=colors)
    plt.title(f"Top 10 Significant Features ({group_name})", pad=15)
    plt.xlabel("Correlation Coefficient", labelpad=10)
    plt.ylabel("")
    plt.axvline(0, color='gray', linestyle='--')

    # 新增保存参数
    plt.savefig(
        f"Correlation_{group_name}.png",
        dpi=300,
        bbox_inches='tight',  # 自动裁剪白边
        pad_inches=0.3,  # 边距微调
        facecolor='white'  # 统一背景色
    )
    plt.close()  # 防止内存泄漏

    return significant


# 执行分析并保存
print("全职群体分析:")
full_time_sig = analyze_group(full_time_df, "Full-time")

print("\n兼职群体分析:")
part_time_sig = analyze_group(part_time_df, "Part-time")

overall_sig = analyze_group(encoded_df, "Overall")

# 输出结果概览
print("\n全职显著特征示例:")
print(full_time_sig[['Feature', 'Correlation']].head(10).to_string(index=False))

print("\n兼职显著特征示例:")
print(part_time_sig[['Feature', 'Correlation']].head(10).to_string(index=False))

print("\n整体显著特征示例:")
print(overall_sig[['Feature', 'Correlation']].head(10).to_string(index=False))




# 导出数据形成表格（Top10）
full_time_sig[['Feature', 'Correlation', 'P-value']].head(10).to_excel("Correlation_Fulltime_Top10.xlsx", index=False)

part_time_sig[['Feature', 'Correlation', 'P-value']].head(10).to_excel("Correlation_Parttime_Top10.xlsx", index=False)

overall_sig[['Feature', 'Correlation', 'P-value']].head(10).to_excel("Correlation_Overall_Top10.xlsx", index=False)

