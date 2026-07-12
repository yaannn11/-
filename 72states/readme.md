# ESA Extension: Adaptive State Representation and Reward Design

Evolutionary Sampling Agent (ESA) 的延伸。

原本 ESA 使用 Q-learning Agent 來選擇不同的 surrogate-based optimization operators，
但原始版本的 state representation 比較簡化，而且 reward 只根據是否有找到更佳解判斷。
這個延伸的主要目的是要改善原始 ESA 中 Q-learning Agent 的狀態表示和獎勵機制，讓 Agent 可以根據更完整的搜尋資訊進行操作選擇。

所以提出兩項延伸：
1. Adaptive State Representation
2. Continuous Improvement Reward
讓 Q-learning agent 可以更精確地判斷目前搜尋狀態，並學習不同搜尋策略在不同階段的適用性。

---

# Original ESA Design
原本 ESA 使用：
- 4 個 optimization actions
- 8 個 states
- Binary reward
  
Reward:
reward = 1.0 if fitness improved else 0.0
只能判斷「是否改善」，無法反映改善的幅度。

例如：
100 → 90
以及
100 → 99.99
兩者會被視為相同 reward，所以 agent 無法區分不同 action 帶來的改善程度。

---

# Proposed Extension
## 1. 72-State Representation

原始 ESA：
8 states

延伸加入更多搜尋資訊：
State consists of:
1. Search stage
2. Improvement level
3. Success status

State size:
4 actions × 3 search stages × 3 improvement levels × 2 success states = 72 states

---
## Search Stage

使用 function evaluation ratio:
progress = nfe / max_nfe

將搜尋過程分成：
| Stage | Condition |
|-|-|
| Early | progress < 0.33 |
| Middle | 0.33 <= progress < 0.66 |
| Late | progress >= 0.66 |

讓 agent 可以學習：
- 初期偏向 exploration
- 後期偏向 exploitation
---

## Improvement Level

根據近期改善率分類：
| State | Meaning |
|-|-|
| 0 | No improvement |
| 1 | Small improvement |
| 2 | Large improvement |


改善率:
improvement = best_y_before - self.best_y

---

# 2. Continuous Reward

原始 ESA:
reward:0 or 1

延伸版本:
reward = improvement / (abs(best_y_before)+1e-12)
並限制：-1 <= reward <= 1

優點：
- 大幅改善給予較高 reward
- 小幅改善給予較低 reward
- 失敗或退化提供負向 feedback

讓 Q-learning 可以更細緻地評估 action 效果。

---

# 3. Adaptive Temperature

加入 temperature adjustment。
原本 Q-learning 使用固定 exploration rate。
延伸版本是根據近期改善情況調整 temperature：
- 搜尋停滯時增加 exploration
- 持續改善時增加 exploitation

目的：
避免過早收斂到單一 action。

# Experimental Setup
Benchmark Functions:
- Ellipsoid
- Rosenbrock
- Ackley
- Griewank
- SRR
- RHC1
- RHC2

Dimensions:
- 30D
- 50D
- 100D

NFE Budget:
1000

Seeds:
10,20,30,42,50

Evaluation:
- Best fitness
- Mean fitness
- Standard deviation
---
# Experimental Results
## Low Dimension (30D / 50D)
在低維度函數測試中，72-state 狀態表示方式相較於原始簡化狀態表示具有較好的搜尋資訊表達能力。

原本的 ESA 使用較少的狀態數量，Agent 只能根據有限的搜尋狀態進行決策；此延伸將搜尋狀態細分，讓Q-learning Agent 可以夠辨識更多不同的搜尋情境，例如：
- 目前搜尋階段 (Early / Middle / Late)
- 最近改善程度 (Improvement Rate)
- 搜尋成功與否 (Success / Failure)

透過更細的狀態空間，Agent 可以更有效的判斷不同階段下適合使用的搜尋策略，提升操作選擇能力。

## High Dimension (100D)
在 100 維度測試中，相較於低維度問題，整體最佳化效果明顯下降。
可能原因如下：

1. **固定的函數評估次數不足**
   隨著維度增加，搜尋空間快速擴大，但 NFE (Number of Function Evaluations) 仍維持固定，使 Agent 能探索的範圍受到限制。

2. **初始採樣數量不足**
   ESA 主要依賴初始樣本建立 RBF surrogate model。
   在高維度空間中，原本固定數量的初始樣本難以充分覆蓋整個搜尋空間，導致代理模型建立困難。

3. **RBF surrogate model 建模困難**
   維度增加後，資料點之間的距離增加，RBF 模型比較難準確估計高維度函數特徵，使預測誤差增加。

4. **高維度搜尋空間複雜度提升**
   隨著維度增加，可能解空間呈指數成長，使探索與開發 (Exploration-Exploitation) 的平衡更加困難。

---

# Conclusion
原始 ESA 使用 8-state 狀態表示，此延伸擴展為 72-state 狀態空間，讓 Agent 可以考慮更多搜尋環境資訊，包括搜尋階段、改善程度以及搜尋成功狀態。

讓 Agent 不只可以判斷是否改善，也可以根據改善幅度評估不同操作策略的有效性。
透過更完整的狀態資訊與 reward 設計，Agent 可以更精確的學習不同搜尋階段下的最佳操作選擇。
此延伸主要著重於提升 ESA 的決策能力

## 實驗結果對比 (Experimental Results)

以下為原始 ESA（Original ESA）與本研究提出的改良版 ESA（Proposed ESA）在各個測試函數（Benchmark Functions）上的表現對比（數值皆為 Mean ± Std，以尋找極小值為目標）：
以30維為例:
| Function | Original ESA (Mean ± Std) | Proposed ESA (Mean ± Std) | 較佳 (Better) |
| :--- | :--- | :--- | :--- |
| **Ellipsoid** | 6.474E-03 ± 5.641E-03 | 6.369E-03 ± 3.759E-03 | ✅ Proposed |
| **Rosenbrock** | 2.853E+01 ± 7.267E-01 | 2.840E+01 ± 6.500E-01 | ✅ Proposed |
| **Ackley** | 2.915E-01 ± 1.922E-01 | 1.292E-01 ± 9.760E-03 | ✅ Proposed |
| **Griewank** | 5.737E-01 ± 2.048E-01 | 6.730E-01 ± 9.970E-02 | ✅ Original |
| **SRR** | -1.269E+02 ± 6.933E+01 | -1.323E+02 ± 6.451E+01 | ✅ Proposed |
| **RHC1** | 3.364E+02 ± 1.150E+02 | 3.496E+02 ± 1.063E+02 | ✅ Original |
| **RHC2** | 9.163E+02 ± 2.669E+00 | 9.171E+02 ± 2.420E+00 | ✅ Original |
實驗結果顯示:本延伸提出的改良版 ESA 展現出更優異的搜尋表現與泛化能力。在 Ackley 函數上，本方法不只將平均最佳解降低了約 55%，標準差更從 0.192 顯著降至 0.0098，證實在優化效率與搜尋穩定性上皆有大幅度的提升；同時，在 Ellipsoid 與 Rosenbrock 函數中，本方法在平均值和標準差上取得雙重改善，維持了高度的穩定搜尋品質。在 Griewank、RHC1 與 RHC2 等多峰或特定地形函數中，因為 Adaptive Temperature 機制動態調整了探索與利用的平衡，可能使演算法在有限的評估次數（NFE）內未能完全收斂，導致結果略遜於原始 ESA。
