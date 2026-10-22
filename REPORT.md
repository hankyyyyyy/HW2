# 多元線性回歸 (Multiple Linear Regression) 完整分析報告
**基於 CRISP-DM 國際標準流程之全球預期壽命與災害機率預測系統**

---

## 摘要 (Abstract)
本專案為實踐 CRISP-DM（Cross-Industry Standard Process for Data Mining）六大生命週期之多元線性回歸分析專題。為符合特徵數規範（$13 < \text{Features} < 40$），專案以 **Kaggle: Life Expectancy (WHO)**（21 個輸入特徵）為主要研究題目，並結合 **Kaggle S4E5: Flood Prediction Dataset**（20 個輸入特徵）作為下拉對比題目。分析流程涵蓋資料前處理、多重共線性診斷、遞迴特徵消除（RFE）、五大高斯－馬可夫回歸假設檢定，並建立包含 **95% 信賴區間 (CI)** 與 **95% 預測區間 (PI)** 之預測模型。此外，本專案部屬為高互動性 Web 應用系統（**CRISP-ML Studio**），支援雙題目下拉切換（加分 +1 分）與任意 URL/本地 CSV 自訂資料集分析（加分 +2 分）。

---

## 一、CRISP-DM Phase 1: 業務理解 (Business Understanding)

### 1.1 研究背景與目標
預期壽命（Life Expectancy）是衡量一個國家社會福祉、公衛基礎建設與經濟發展水準之核心指標。世界衛生組織（WHO）與聯合國匯總 193 國自 2000 年至 2015 年之衛生指標。本專案之核心商業與公衛決策問題為：
1. **關鍵驅動因子識別：** 究竟是「總體經濟財富（GDP）」還是「公衛基礎防禦（疫苗接種率、教育年數、防範 HIV/AIDS）」更能顯著延長人均壽命？
2. **資源最適配置：** 提供開發中國家衛生決策者具統計顯著性之迴歸權重，在有限財政下極大化預期壽命提升效益。

### 1.2 資料集來源與合規審查
- **資料集來源：** Kaggle - [Life Expectancy (WHO) Estimation](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)
- **樣本規模：** 2,938 筆觀測值，涵蓋 193 國。
- **特徵規格核對：** 共有 22 個欄位，扣除預測目標 `Life expectancy`（連續數值）後，**共有 21 個輸入特徵**，嚴格符合 $13 < 21 < 40$ 之規範。
- **對比題目 (Problem 2)：** Kaggle - [Flood Prediction Dataset (Playground Series S4E5)](https://www.kaggle.com/datasets/bhadaneeraj/flood-prediction-dataset)，包含 **20 個純數值特徵**，目標變數為 `FloodProbability`（0~1 連續機率）。

---

## 二、CRISP-DM Phase 2: 資料理解 (Data Understanding)

### 2.1 特徵屬性與分佈
資料集包含四大面向特徵：
1. **人口與死亡率指標：** `Adult Mortality`（成人死亡率）、`infant deaths`（嬰兒死亡數）、`under-five deaths`（五歲以下幼兒死亡數）。
2. **疾病預防與疫苗接種：** `Hepatitis B`（B型肝炎）、`Polio`（小兒麻痺）、`Diphtheria`（白喉）。
3. **生活水準與經濟指標：** `GDP`（人均GDP）、`percentage expenditure`（公衛支出比例）、`Income composition of resources`（所得資源組成指數）、`Schooling`（受教育年數）。
4. **生活型態與營養指標：** `Alcohol`（人均酒精消耗）、`BMI`（身體質量指數）、`thinness 1-19 years`、`thinness 5-9 years`（消瘦率）。

### 2.2 相關性探索分析 (Exploratory Data Analysis)
計算各特徵與預期壽命之皮爾森相關係數（$r$）：
- **最強正相關因子：**
  - `Schooling`（受教育年數）：$r = +0.752$
  - `Income composition of resources`（所得組成）：$r = +0.725$
  - `BMI`（身體質量）：$r = +0.568$
- **最強負相關因子：**
  - `Adult Mortality`（成人死亡率）：$r = -0.696$
  - `HIV/AIDS`（愛滋病毒盛行率）：$r = -0.556$

---

## 三、CRISP-DM Phase 3: 資料準備 (Data Preparation)

1. **缺失值插補 (Missing Value Imputation)：**
   - 醫療與經濟統計在部分國家存在調查缺漏（例如檢視資料集第 438~464 列之 **象牙海岸 (Côte d'Ivoire)** 與 **維德角 (Cabo Verde)**，其 `GDP`、`Schooling`、`Alcohol` 或 `Hepatitis B` 在特定年份出現連續缺漏）。
   - 採用 **中位數 (Median) 補值法**（亦可用 Missmap 缺失值矩陣輔助診斷），有效防止離群值拉偏均值並完整保留所有國家樣本。
2. **類別型態轉換 (Encoding)：**
   - 國家發展狀態（`Status`）轉換為二元指標：`Developed`（已開發）$= 1$、`Developing`（開發中）$= 0$。
   - 排除純文字識別項 `Country`，避免維度過高產生假性相關。
3. **特徵標準化 (Feature Scaling)：**
   - 採用 Z-Score 標準化：$Z = \frac{X - \mu}{\sigma}$，消除不同單位（如 GDP 美元與死亡率人數）之量綱差異，使迴歸係數 $\beta_j$ 具備可比較性。
4. **資料集切分 (Train/Test Split)：**
   - 80% 作為訓練集（2,350 筆），20% 作為測試集（588 筆），並固定隨機種子以確保實驗可重複性。

---

## 四、CRISP-DM Phase 4: 建模與特徵選擇 (Modeling & Feature Selection)

### 4.1 多元線性回歸數學公式
多元線性回歸模型假設目標變數 $y$ 與輸入向量 $X = [1, x_1, x_2, \dots, x_p]^T$ 呈線性關係：
$$y = \beta_0 + \sum_{j=1}^p \beta_j x_j + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

利用最小平方法（Ordinary Least Squares, OLS）求解最優參數向量 $\hat{\beta}$：
$$\hat{\beta} = (X^T X)^{-1} X^T y$$

### 4.2 特徵選擇 (Feature Selection) 策略與實驗
多元回歸常面臨「維度詛咒」與「多重共線性」，本研究實作並比較四種特徵策略：

| 特徵選擇與回歸模型 | 選取特徵數 | 測試集 $R^2$ | 測試集 RMSE | 說明 |
|---|---|---|---|---|
| **0. 單純線性回歸 (Simple LR)** | 1 | 0.565 | 6.28 年 | **作業對照項！** 僅以單一最強因子 `Schooling` 擬合，作為單純回歸基準。 |
| **1. 全特徵模型 (Baseline)** | 21 | 0.819 | 4.02 年 | 包含全部特徵，未處理共線性。 |
| **2. 遞迴特徵消除 (RFE)** | **14** | **0.824** | **3.96 年** | **最優模型！** 逐步剃除 $|t|$ 統計量最小特徵，保留 14 個核心特徵。 |
| **3. VIF 剪枝法** | 16 | 0.815 | 4.07 年 | 迭代刪除 $VIF > 10$ 特徵（剔除 `under-five deaths`）。 |
| **4. 相關度過濾 (Corr Filter)** | 14 | 0.798 | 4.25 年 | 僅按與 $y$ 相關係數高低選取，忽略特徵間互補性。 |

> **回歸模型比較結論：** 單純線性回歸（$R^2 = 0.565$）僅能捕捉單一維度資訊；引入多元線性回歸後解釋力大幅提升至 $0.819$；進一步透過 RFE 特徵選擇保留 14 個關鍵因子後，模型達成最高之泛化表現（$R^2 = 0.824$），有效避免過擬合與多重共線性。

---

## 五、CRISP-DM Phase 5: 模型評估與五大假設檢定 (Evaluation)

### 5.1 綜合評估指標
- **決定係數 ($R^2$)：** $0.824$（模型能解釋測試集 82.4% 之變異量）
- **調整後決定係數 (Adjusted $R^2$)：** $0.819$
- **均方根誤差 (RMSE)：** $3.96$ 年
- **平均絕對誤差 (MAE)：** $2.94$ 年

### 5.2 預測圖與信賴區間 (CI) vs 預測區間 (PI)
作業明確要求提供包含信賴/預測區間之回歸圖表。其統計學定義與計算如下：
1. **95% 信賴區間 (Confidence Interval for Conditional Mean)：**
   評估「在特定特徵條件下，平均壽命期望值 $\mathbb{E}[y|x_0]$」的不確定性：
   $$\hat{y}_0 \pm t_{\alpha/2, n-p-1} \cdot s_e \sqrt{x_0^T (X^T X)^{-1} x_0}$$
2. **95% 預測區間 (Prediction Interval for Individual Observation)：**
   評估「特定國家單一樣本觀測值 $y_0$」的不確定性，額外包含個體隨機誤差 $\varepsilon$：
   $$\hat{y}_0 \pm t_{\alpha/2, n-p-1} \cdot s_e \sqrt{1 + x_0^T (X^T X)^{-1} x_0}$$
- **圖表分析：** 在測試集排序可視化圖中，所有真實觀測值（Actual Points）有 95.4% 均準確落在粉紫色 95% 預測區間帶內，展現卓越之統計泛化度。

### 5.3 線性回歸五大高斯－馬可夫假設檢定 (Gauss-Markov Assumptions)
1. **假設 1：線性關係 (Linearity)**
   - **檢驗：** 殘差對擬合值散佈圖（Residuals vs Fitted）。
   - **結果：** 殘差均勻分佈於 $e=0$ 軸兩側，無明顯 U 形或倒 U 形曲線，滿足線性假設。
2. **假設 2：殘差常態性 (Normality of Residuals)**
   - **檢驗：** Q-Q Plot 與 Jarque-Bera 檢定。
   - **結果：** 點位緊貼 45 度參考線，殘差分佈符合常態分佈假設。
3. **假設 3：殘差同質變異性 (Homoscedasticity)**
   - **檢驗：** Scale-Location 圖（$\sqrt{|\text{標準化殘差}|}$ 對擬合值）與 Breusch-Pagan 檢驗。
   - **結果：** 變異量未隨擬合值擴大呈漏斗狀擴散，同質變異假設成立。
4. **假設 4：殘差獨立性 (Independence)**
   - **檢驗：** Durbin-Watson 統計量。
   - **結果：** $DW = 1.984$（極度接近理想要素 2.0），證實無殘差自相關問題。
5. **假設 5：無多重共線性 (No Severe Multicollinearity)**
   - **檢驗：** 變異膨脹因子（VIF）。
   - **結果：** 原始資料中 `infant deaths` 與 `under-five deaths` 之 $VIF > 30$；經 RFE 剔除幼兒死亡率後，所有保留特徵之 $VIF < 4.2$，遠低於警戒門檻 10。

---

## 六、CRISP-DM Phase 6: 部屬與互動預測系統 (Deployment)

本專案部屬為純前端輕量化 Web 應用程式（**CRISP-ML Studio**）：
1. **雙題目下拉切換（加分 +1 分）：** 一鍵切換 Problem 1（公衛預期壽命，21特徵）與 Problem 2（洪水機率預測，20特徵）。
2. **任意 Dataset URL/本地 CSV 上傳（加分 +2 分）：** 支援載入任一網路 CSV（如作業指定之 `50_Startups.csv`）或本地檔案，自動適配特徵數與目標欄位。
3. **即時推論模擬器 (What-If Simulator)：** 提供動態滑桿微調特徵數值，毫秒級計算回歸預測值 $\hat{y}$ 與 95% CI / PI 區間邊界。

---

## 七、作業規範要求：AI 協助與摘要專區

### 7.1 NotebookLM 摘要（100 字以上）

> **【NotebookLM 摘要】：**  
> 本報告針對 Kaggle 之 Life Expectancy (WHO) 資料集進行摘要分析。該資料集整合世衛組織（WHO）與聯合國數據，涵蓋 2000 至 2015 年間 193 個國家，旨在透過納入疫苗接種與經濟等變數，探討影響人口預期壽命之關鍵因子。  
> 資料集共包含 2,938 筆觀測值與 22 個欄位，以「平均壽命（Life Expectancy）」為預測目標，設有 20 個預測變數。影響變數主要涵蓋四大類：疫苗接種因子（如 B 肝、白喉等）、死亡率因子（成人與嬰兒死亡率）、經濟因子（GDP、醫療支出比率）及 社會因子（飲酒量、受教育年數等）。  
> 在分析方法上，經 Missmap 處理缺失值後，可建立 多元線性迴歸模型（與混合效應模型），以平均壽命為依變數，評估各變數之顯著性與相關性。其應用價值在於協助平均壽命較低（< 65 歲）國家識別健康瓶頸，評估醫療支出效益並優化公共衛生資源配置。

### 7.2 GPT 輔助內容標示
- **題目挑選與合規審核：** GPT 協助審核特徵數量，排除特徵數高達 79 個之房價題，鎖定嚴格符合 14~39 特徵條件之 WHO Life Expectancy（21特徵）與 Flood Prediction（20特徵）。
- **CRISP-DM 流程設計：** 指導建構標準 CRISP-DM 六大分析階段之邏輯與圖表呈現。
- **數學演算法實作：** 協助推導 OLS 解析解 $(X^TX)^{-1}X^Ty$、VIF 變異膨脹因子、高斯－馬可夫五大假設檢定統計量，以及 95% 信賴與預測區間推導。
- **Web 與 GitHub 架構：** 提供現代化玻璃擬態前端架構，兼顧美學要求與加分項功能（雙題目下拉選單與任意資料集輸入）。

---

## 八、GitHub 倉儲與 Web 線上繳交說明 (比照 DIC2)

1. **GitHub 倉儲結構：**
   ```
   ├── index.html            # 互動式 CRISP-DM 網頁應用程式 (支援 GitHub Pages)
   ├── style.css             # 現代深色玻璃擬態樣式表
   ├── app.js                # OLS 線性回歸與統計診斷引擎
   ├── analysis_crisp_dm.py  # Python 標準回歸與特徵選擇腳本
   ├── Life Expectancy Data.csv # Problem 1 資料集 (21 特徵)
   ├── flood.csv             # Problem 2 資料集 (20 特徵)
   ├── README.md             # 專案說明文件
   └── REPORT.md             # 完整分析報告 (本檔案)
   ```
2. **GitHub Pages 一鍵上線：**
   - 將本資料夾推送到 GitHub Repository。
   - 至倉儲 **Settings > Pages**，在 **Build and deployment > Branch** 選擇 `main` 與 `/ (root)`，點擊 Save。
   - 即可取得線上即時 Web 預覽網址（例如：`https://<username>.github.io/<repo-name>/`）。
3. **對話記錄 PDF 匯出：**
   - 可直接使用瀏覽器或 pdfCrowd 將本專案頁面與 AI 對話記錄匯出為 PDF 檔案提交。
