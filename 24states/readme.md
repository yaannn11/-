## 🚀 專案改進內容

本專案以 **Evolutionary Sampling Agent（ESA）** 為基礎進行改良，針對 **Q-learning 決策機制**與**探索策略（Exploration）**進行優化，以提升演算法在高維最佳化問題中的搜尋效率、收斂速度與穩定性。

---

### 1️⃣ Adaptive Temperature 機制

原始 ESA 在 Q-learning 的 Action Selection 階段採用固定的 **Temperature (T)**，因此整個搜尋過程中的探索（Exploration）與利用（Exploitation）比例維持不變。

本研究導入 **Adaptive Temperature** 機制，根據近期搜尋表現（Improvement Rate）動態調整 Temperature。

- 當 **Improvement Rate 較低**時，表示搜尋可能陷入停滯，因此提高 Temperature，增加探索（Exploration）的機率，使 Agent 嘗試更多不同的搜尋策略。
- 當 **Improvement Rate 較高**時，表示目前搜尋方向具有良好效果，因此降低 Temperature，使 Agent 更傾向利用（Exploitation）目前表現較佳的 Action。

透過此機制，ESA 能依據搜尋狀態自動調整探索與利用之間的平衡，而非依賴固定參數。

---

### 2️⃣ Improvement-Rate-Based 24-State Representation

原始 ESA 的 Q-learning 僅使用較少的狀態（State）資訊，因此無法充分反映目前搜尋品質與演算法的收斂情形。

本研究加入 **Improvement Rate** 作為新的狀態資訊，將 Q-learning 的狀態空間擴充為 **24 個 States**。

每個 State 由下列三項資訊組成：

- **Action（4 種）**：ESA 所提供的四種搜尋策略（A1～A4）
- **Improvement Level（3 種）**：根據近期 Improvement Rate 分為 Low、Medium、High
- **Success Status（2 種）**：本次搜尋是否成功改善最佳解（Success / Failure）

因此總狀態數為：

> **4 Actions × 3 Improvement Levels × 2 Success Status = 24 States**

擴充後的 State Representation 不僅能描述目前所採用的搜尋策略，也能反映搜尋是否處於停滯或持續改善的狀態，使 Q-learning 能做出更精確的 Action Selection。

---

### 3️⃣ Improvement Rate 計算方式

本研究利用近期搜尋結果計算 **Improvement Rate**，用以衡量目前搜尋品質。

若本次評估找到新的最佳解，則記錄為 **1**；否則記錄為 **0**。

接著利用最近一段搜尋紀錄計算移動平均（Moving Average），作為目前的 Improvement Rate。

並依照下列門檻將其離散化：

| Improvement Rate | Improvement Level |
|------------------|-------------------|
| < 0.1 | Low |
| 0.1 ～ 0.5 | Medium |
| ≥ 0.5 | High |

離散化後的 Improvement Level 同時應用於：

1. Adaptive Temperature 的調整
2. Q-learning 的 State Representation

讓探索策略與狀態表示採用一致的判斷依據。

---

### 4️⃣ 預期效益

本研究透過 **Adaptive Temperature** 與 **24-State Q-learning** 的結合，希望達成以下目標：

- 提升 Exploration 與 Exploitation 的平衡能力。
- 降低演算法陷入局部最佳解（Local Optimum）的機率。
- 提高不同 Benchmark Functions 上的搜尋穩定性。
- 提升高維最佳化問題的收斂效率。
- 強化 Q-learning 在不同搜尋階段的決策能力。

---

### 📌 改進重點總覽

| 改進項目 | 原始 ESA | 本研究 |
|-----------|----------|---------|
| Temperature | 固定 | Adaptive Temperature |
| State 數量 | 原始 State | 24 States |
| State 資訊 | Action、Success | Action、Improvement Level、Success |
| Exploration / Exploitation | 固定比例 | 動態調整 |
| Improvement Rate | 未使用 | State + Temperature 共同使用 |
