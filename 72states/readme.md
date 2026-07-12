# ESA Extension: Adaptive State Representation and Reward Design

此檔案為 Evolutionary Sampling Agent (ESA) 的延伸版本。

原始 ESA 使用 Q-learning Agent 來選擇不同的 surrogate-based optimization operators，
但原始版本的 state representation 較簡化，且 reward 僅依據是否找到更佳解判斷。

本研究提出兩項延伸：

1. Adaptive State Representation
2. Continuous Improvement Reward

讓 Q-learning agent 能更精確地判斷目前搜尋狀態，並學習不同搜尋策略在不同階段的適用性。

---

# Original ESA Design

原始 ESA 使用：
- 4 個 optimization actions
- 8 個 states
- Binary reward
- 
Reward:
reward = 1.0 if fitness improved else 0.0
只能判斷「是否改善」，
無法反映改善幅度。

例如：

100 → 90
以及
100 → 99.99

兩者皆被視為相同 reward。
所以 agent 無法區分不同 action 帶來的改善程度。

---

# Proposed Extension

## 1. 72-State Representation

原始 ESA：
8 states

延伸版本加入更多搜尋資訊：

State consists of:

1. Search stage
2. Improvement level
3. Success status
4. Selected action


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
improvement_rate =
(best_y_before - best_y)/(abs(best_y_before)+epsilon)

---

# 2. Continuous Reward

原始 ESA:
reward:0 or 1


延伸版本:
reward:(best_y_before - best_y)/(abs(best_y_before)+epsilon)
並限制：
-1 <= reward <= 1


優點：
- 大幅改善給予較高 reward
- 小幅改善給予較低 reward
- 失敗或退化提供負向 feedback

使 Q-learning 可以更細緻地評估 action 效果。

---

# 3. Adaptive Temperature

另外加入 temperature adjustment。

原始 Q-learning 使用固定 exploration rate。

延伸版本根據近期改善情況調整 temperature：
- 搜尋停滯時增加 exploration
- 持續改善時增加 exploitation


目的：
避免過早收斂到單一 action。

---

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

72-state representation provides richer search information.

Compared with original simplified state representation,
the agent can distinguish different search situations and select operators more effectively.

---

## High Dimension (100D)

Performance decreases compared with lower dimensions.

Possible reasons:

1. Fixed evaluation budget becomes insufficient.
2. Initial sampling size is limited.
3. RBF surrogate becomes harder to approximate in high-dimensional space.
4. Search space grows exponentially with dimension.

Future improvement:

- Increase initialization samples
- Adaptive surrogate model selection
- Dimension-aware parameter adjustment


---

# Conclusion

This extension improves ESA's decision mechanism by extending the original 8-state Q-learning model into a 72-state adaptive representation.

By combining continuous reward feedback and adaptive exploration control,
the agent can better evaluate operator effectiveness under different optimization stages.

The proposed method focuses on improving ESA's decision-making capability rather than modifying the optimization operators themselves.
