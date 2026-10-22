# CRISP-ML Studio: Multiple Linear Regression (14–39 Features)

> **基於 CRISP-DM 六大階段之多元線性回歸分析與互動式統計診斷平台**  
> 專為符合「**13 < 輸入特徵數 < 40**」規範與 Kaggle 公開資料集回歸作業設計。

[![Features Compliant](https://img.shields.io/badge/Features-21%20Compliant%20(14--39)-blue.svg)](#)
[![CRISP-DM Standard](https://img.shields.io/badge/Process-CRISP--DM%20Phase%201~6-green.svg)](#)
[![Bonus Points](https://img.shields.io/badge/Bonus-Dual%20Problems%20(%2B1)%20%2B%20Custom%20URL%20(%2B2)-gold.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#)

---

## 🌟 作業要求與加分項落實清單

- [x] **資料集來源：** Kaggle 公開資料集 [Life Expectancy (WHO)](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who)，輸入特徵數為 **21 個**，完全落在 $14 \le \text{Features} \le 39$ 範圍內。
- [x] **分析模型：** 多元線性回歸（Multiple Linear Regression, OLS 解析解 $\hat{\beta} = (X^TX)^{-1}X^Ty$）。
- [x] **特徵選擇 (Feature Selection)：** 實作並比較全特徵 (Baseline)、遞迴特徵消除 (RFE)、VIF 共線性剔除與皮爾森相關係數篩選。
- [x] **模型評估 (Model Evaluation)：** 完整輸出 MSE, RMSE, MAE, $R^2$ 與 Adjusted $R^2$。
- [x] **預測圖與區間帶：** 提供測試集擬合預測圖，精確繪製 **95% 平均值信賴區間 (CI)** 與 **95% 個體預測區間 (PI)** 陰影帶。
- [x] **五大高斯－馬可夫回歸假設檢定：**
  1. 線性關係 (Linearity) - 殘差對擬合值散佈圖
  2. 殘差常態性 (Normality) - Q-Q Plot 與常態分佈分位數
  3. 同質變異性 (Homoscedasticity) - Scale-Location 圖
  4. 殘差獨立性 (Independence) - Durbin-Watson 統計量 ($\approx 1.98$)
  5. 無嚴重多重共線性 (Multicollinearity) - VIF 變異膨脹因子檢定
- [x] **加分項 1（加總分 1分）：** 下拉視窗呈現兩種 Problem：
  - 🏥 **Problem 1:** WHO 預期壽命預測 (21 Features, 2,938 Rows)
  - 🌊 **Problem 2:** Kaggle S4E5 洪水機率預測 (20 Features, 300+ Rows)
- [x] **加分項 2（加總分 2分）：** 可任意輸入 Dataset：
  - 支援輸入任何公開 CSV 網址（預設包含作業範例 `50_Startups.csv`）
  - 支援本地 CSV 檔案上傳
- [x] **AI 協助規範：** 報告中明確標示「GPT 輔助內容」與 100 字以上「NotebookLM 摘要」。
- [x] **繳交格式：** 比照 DIC2，同時提供 GitHub 倉儲程式碼與 Web 線上靜態應用（GitHub Pages）。

---

## 🚀 快速啟動 (Quick Start)

### 方式一：直接在瀏覽器開啟 (無需安裝任何套件)
雙擊開啟專案目錄下的 `index.html` 即可使用完整互動系統！

### 方式二：使用本地 HTTP 伺服器
若希望獲得最佳靜態載入體驗：
```powershell
# 使用內建之 PowerShell 輕量伺服器
powershell -ExecutionPolicy Bypass -File .\server.ps1 -Port 8080
```
或使用 Python（若有安裝）：
```bash
python -m http.server 8080
```
在瀏覽器輸入 `http://localhost:8080` 即可進入系統。

### 方式三：執行 Python 回歸腳本
```bash
python analysis_crisp_dm.py
```

---

## 🌐 GitHub Pages 部署步驟 (Web 繳交方式)

1. 在 GitHub 上建立一個新的公開倉儲 (Public Repository)。
2. 將本專案所有檔案推送至該倉儲：
   ```bash
   git init
   git add .
   git commit -m "feat: complete CRISP-DM Multiple Linear Regression project"
   git branch -M main
   git remote add origin https://github.com/<你的使用者名稱>/<你的倉儲名稱>.git
   git push -u origin main
   ```
3. 進入 GitHub 倉儲的 **Settings > Pages**。
4. 在 **Build and deployment > Branch** 選擇 `main` 與 `/ (root)`，點擊 **Save**。
5. 等待 1~2 分鐘，即可獲得線上預覽網址（例如：`https://<username>.github.io/<repo-name>/`），直接複製填寫於作業繳交表單！

---

## 📁 檔案架構

| 檔案名稱 | 說明 |
|---|---|
| `index.html` | CRISP-ML Studio 互動式主網頁 (支援 CRISP-DM 6 階段導覽) |
| `style.css` | 現代玻璃擬態深色主題樣式表 |
| `app.js` | 矩陣運算、OLS 線性回歸、特徵選擇、五大假設檢定與 Chart.js 視覺化引擎 |
| `analysis_crisp_dm.py` | Python 完整回歸腳本 (scikit-learn + statsmodels) |
| `Life Expectancy Data.csv` | Kaggle WHO 預期壽命資料集 (21 輸入特徵) |
| `flood.csv` | Kaggle 洪水機率預測資料集 (20 輸入特徵，供下拉選單切換) |
| `REPORT.md` | 詳細學術分析報告書 (含 NotebookLM 摘要與 GPT 輔助說明) |
| `server.ps1` | 本地靜態 HTTP 伺服器啟動腳本 |

---

## 🤖 作業要求：AI 協助與摘要專區

### 【NotebookLM 摘要】（100 字以上）
> 本報告針對 Kaggle 之 Life Expectancy (WHO) 資料集進行摘要分析。該資料集整合世衛組織（WHO）與聯合國數據，涵蓋 2000 至 2015 年間 193 個國家，旨在透過納入疫苗接種與經濟等變數，探討影響人口預期壽命之關鍵因子。  
> 資料集共包含 2,938 筆觀測值與 22 個欄位，以「平均壽命（Life Expectancy）」為預測目標，設有 20 個預測變數。影響變數主要涵蓋四大類：疫苗接種因子（如 B 肝、白喉等）、死亡率因子（成人與嬰兒死亡率）、經濟因子（GDP、醫療支出比率）及 社會因子（飲酒量、受教育年數等）。  
> 在分析方法上，經 Missmap 處理缺失值後，可建立 多元線性迴歸模型（與混合效應模型），以平均壽命為依變數，評估各變數之顯著性與相關性。其應用價值在於協助平均壽命較低（< 65 歲）國家識別健康瓶頸，評估醫療支出效益並優化公共衛生資源配置。

### 【GPT 輔助內容】
- 題目挑選與 14–39 特徵合規性檢驗
- CRISP-DM 六大分析流程框架搭建
- OLS 矩陣公式求解、RFE 特徵選擇、VIF 共線性檢定及 95% 信賴/預測區間推導
- 現代前端網頁系統建置（支援雙題目切換 +1分、任意輸入資料集 +2分）

