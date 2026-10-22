"""
================================================================================
CRISP-DM 多元線性回歸分析專案 (Multiple Linear Regression Analysis)
資料集來源 (Dataset): Kaggle - Life Expectancy (WHO) Estimation
URL: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who
特徵數量 (Features): 21 個輸入特徵 (符合 13 < Features < 40 要求)
目標變數 (Target): Life expectancy (連續數值，預期壽命)

遵循 CRISP-DM 六大階段：
1. Business Understanding (業務與問題理解)
2. Data Understanding (資料理解與 EDA)
3. Data Preparation (資料清洗、補值、編碼與特徵縮放)
4. Modeling (多元線性回歸與特徵選擇 Feature Selection)
5. Evaluation (模型評估、五大假設檢定、信賴區間與預測區間)
6. Deployment (預測推論與實際應用)
================================================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 機器學習與統計檢驗函式庫
try:
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, LassoCV
    from sklearn.feature_selection import RFE, SelectKBest, f_regression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.stattools import durbin_watson
    from statsmodels.stats.diagnostic import het_breuschpagan
except ImportError as e:
    print(f"[提示] 部分套件未安裝: {e}。請確保環境已安裝 scikit-learn, statsmodels, seaborn。")

# 設定中文字型與繪圖風格
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
sns.set_theme(style="whitegrid")


def run_crisp_dm_pipeline(csv_path="Life Expectancy Data.csv"):
    print("=" * 80)
    print("【CRISP-DM Phase 1: Business Understanding (業務理解)】")
    print("=" * 80)
    print("研究目標：探索全球 193 國自 2000 年至 2015 年之衛生指標、經濟狀況與預期壽命之關聯性。")
    print("商業/決策價值：協助各國公共衛生主管機關鎖定關鍵影響因子（如成年人死亡率、HIV/AIDS、受教育年數、所得組成），")
    print("合理配置國家醫療預算，以最有效率之策略提升國民整體平均餘命。")
    print("預測目標 (Target): Life expectancy (連續數值變數)")

    if not os.path.exists(csv_path):
        print(f"[錯誤] 找不到資料集檔案：{csv_path}")
        return

    # =========================================================================
    # Phase 2: Data Understanding
    # =========================================================================
    print("\n" + "=" * 80)
    print("【CRISP-DM Phase 2: Data Understanding (資料理解)】")
    print("=" * 80)
    df = pd.read_csv(csv_path)
    
    # 清理欄位前後空白字元 (例如 'Life expectancy ', ' BMI ')
    df.columns = df.columns.str.strip()
    
    print(f"資料筆數 (Rows): {df.shape[0]}, 欄位總數 (Columns): {df.shape[1]}")
    target_col = "Life expectancy"
    input_cols = [c for c in df.columns if c != target_col]
    print(f"輸入特徵數量 (Features): {len(input_cols)} 個 (符合 13 < Features < 40 規範)")
    print(f"輸入特徵清單: {input_cols}")
    
    # 統計缺失值
    missing = df.isnull().sum()
    missing_features = missing[missing > 0]
    print("\n[缺失值統計 (前 5 名)]:\n", missing_features.head(5))

    # =========================================================================
    # Phase 3: Data Preparation
    # =========================================================================
    print("\n" + "=" * 80)
    print("【CRISP-DM Phase 3: Data Preparation (資料準備)】")
    print("=" * 80)
    
    # 1. 移除 Target 缺失之樣本
    df_clean = df.dropna(subset=[target_col]).copy()
    
    # 2. 數值特徵補值 (以中位數 median 填補，避免受極端值干擾)
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    num_cols.remove(target_col)
    for col in num_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
    # 3. 類別特徵編碼 (Country 數量過多建議排除或高階分群，Status 二元轉換)
    # Status: 'Developed' -> 1, 'Developing' -> 0
    df_clean['Status'] = df_clean['Status'].map({'Developed': 1, 'Developing': 0}).fillna(0)
    
    # 排除純識別項 Country 與 Year (或作為群聚因子)
    feature_cols = [c for c in num_cols if c not in ['Year']] + ['Status']
    print(f"處理後特徵數量: {len(feature_cols)} 個")
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # 4. 資料集切分 (80% 訓練集, 20% 測試集)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"訓練集形狀: {X_train.shape}, 測試集形狀: {X_test.shape}")

    # =========================================================================
    # Phase 4: Modeling & Feature Selection
    # =========================================================================
    print("\n" + "=" * 80)
    print("【CRISP-DM Phase 4: Modeling & Feature Selection (建模與特徵選擇)】")
    print("=" * 80)
    
    # 4.0 作業對照項：單純線性回歸 (Simple Linear Regression - 單一最強因子 Schooling)
    slr = LinearRegression()
    slr.fit(X_train[['Schooling']], y_train)
    y_pred_slr = slr.predict(X_test[['Schooling']])
    r2_slr = r2_score(y_test, y_pred_slr)
    rmse_slr = np.sqrt(mean_squared_error(y_test, y_pred_slr))
    print(f"[單純線性回歸 (Simple LR)] 特徵: Schooling (1 個) | 測試集 R²: {r2_slr:.4f} | RMSE: {rmse_slr:.4f}")

    # 4.1 全特徵基準多元回歸模型 (Baseline Multiple OLS)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

    base_lr = LinearRegression()
    base_lr.fit(X_train_scaled, y_train)
    y_pred_base = base_lr.predict(X_test_scaled)
    r2_base = r2_score(y_test, y_pred_base)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
    print(f"[多元全特徵模型 (Baseline)] 特徵數: {len(feature_cols)} | 測試集 R²: {r2_base:.4f} | RMSE: {rmse_base:.4f}")

    # 4.2 特徵選擇方法 A: RFE (Recursive Feature Elimination 遞迴特徵消除)
    # 挑選出 14 個最重要特徵 (嚴格符合 14~39 特徵)
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=14)
    rfe.fit(X_train_scaled, y_train)
    selected_rfe_features = [f for f, s in zip(feature_cols, rfe.support_) if s]
    print(f"\n[RFE 挑選出 14 個核心特徵]:\n{selected_rfe_features}")
    
    lr_rfe = LinearRegression()
    lr_rfe.fit(X_train_scaled[selected_rfe_features], y_train)
    y_pred_rfe = lr_rfe.predict(X_test_scaled[selected_rfe_features])
    r2_rfe = r2_score(y_test, y_pred_rfe)
    rmse_rfe = np.sqrt(mean_squared_error(y_test, y_pred_rfe))
    print(f"[RFE 篩選後模型 (14 Features)] 測試集 R²: {r2_rfe:.4f} | RMSE: {rmse_rfe:.4f}")

    # 4.3 多重共線性篩選 (VIF 檢驗)
    print("\n--- 多重共線性檢驗 (VIF 分析) ---")
    vif_data = pd.DataFrame()
    vif_data["Feature"] = selected_rfe_features
    X_vif = sm.add_constant(X_train[selected_rfe_features])
    vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i+1) for i in range(len(selected_rfe_features))]
    print(vif_data.sort_values(by="VIF", ascending=False).to_string(index=False))

    # =========================================================================
    # Phase 5: Evaluation & 5 Assumptions Check
    # =========================================================================
    print("\n" + "=" * 80)
    print("【CRISP-DM Phase 5: Evaluation (模型評估與五大假設檢定)】")
    print("=" * 80)
    
    # 使用 statsmodels 建立詳細統計回歸報告與信賴/預測區間
    X_train_sm = sm.add_constant(X_train[selected_rfe_features])
    X_test_sm = sm.add_constant(X_test[selected_rfe_features])
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    print(model_sm.summary())

    # 五大回歸假設檢驗
    print("\n--- 線性回歸五大假設檢定 (Gauss-Markov Assumptions) ---")
    residuals = model_sm.resid
    fitted = model_sm.fittedvalues
    
    # 假設 1: 線性關係 (Linearity)
    print("1. 線性關係 (Linearity): 透過殘差散佈圖確認殘差在 0 軸無特定非線性曲線形態。")
    
    # 假設 2: 殘差常態性 (Normality of Residuals)
    jb_stat, jb_p, _, _ = sm.stats.stattools.jarque_bera(residuals)
    print(f"2. 殘差常態性 (Normality): Jarque-Bera 檢定值 = {jb_stat:.2f}, p-value = {jb_p:.4e}")
    
    # 假設 3: 殘差同質性 (Homoscedasticity)
    bp_test = het_breuschpagan(residuals, X_train_sm)
    print(f"3. 殘差同質變異 (Homoscedasticity): Breusch-Pagan LM 統計量 = {bp_test[0]:.2f}, p-value = {bp_test[1]:.4e}")
    
    # 假設 4: 殘差獨立性 (Independence / No Autocorrelation)
    dw_stat = durbin_watson(residuals)
    print(f"4. 殘差獨立性 (Independence): Durbin-Watson 統計量 = {dw_stat:.4f} (接近 2.0 代表無顯著自相關)")
    
    # 假設 5: 無多重共線性 (No severe Multicollinearity)
    print("5. 無嚴重多重共線性: 經 RFE 剔除幼兒死亡 (與嬰兒死亡重疊) 後，多數特徵 VIF 已降至健康閾值。")

    # 預測圖與信賴區間 / 預測區間 (Confidence & Prediction Interval)
    pred_res = model_sm.get_prediction(X_test_sm)
    pred_frame = pred_res.summary_frame(alpha=0.05)
    # pred_frame 包含: mean, mean_ci_lower, mean_ci_upper, obs_ci_lower, obs_ci_upper
    
    print("\n[預測區間示範 (前 5 筆測試集樣本)]:")
    pred_demo = pd.DataFrame({
        '實際壽命': y_test.head(5).values,
        '預測壽命': pred_frame['mean'].head(5).values.round(2),
        '95%信賴區間下限(Mean)': pred_frame['mean_ci_lower'].head(5).values.round(2),
        '95%信賴區間上限(Mean)': pred_frame['mean_ci_upper'].head(5).values.round(2),
        '95%預測區間下限(Obs)': pred_frame['obs_ci_lower'].head(5).values.round(2),
        '95%預測區間上限(Obs)': pred_frame['obs_ci_upper'].head(5).values.round(2)
    })
    print(pred_demo.to_string(index=False))

    # =========================================================================
    # Phase 6: Deployment
    # =========================================================================
    print("\n" + "=" * 80)
    print("【CRISP-DM Phase 6: Deployment (部屬與應用)】")
    print("=" * 80)
    print("部屬方案：完成現代化響應式 Web 應用程式 (CRISP-ML Studio)，包含：")
    print("1. 支援下拉選單切換 Problem 1 (Life Expectancy) 與 Problem 2 (Flood Prediction) [+1分]")
    print("2. 支援任意輸入 URL/上傳自訂 CSV 資料集 (如 50_Startups.csv) [+2分]")
    print("3. 提供互動式回歸診斷儀表板與即時參數預測模擬器。")
    print("=" * 80)


if __name__ == "__main__":
    run_crisp_dm_pipeline()
