# 實驗結果
* 優化性能對比
* <img width="1560" height="804" alt="image" src="https://github.com/user-attachments/assets/d5e63dbd-563d-4f5e-a3d7-5ace10cec2ee" />
* 自適應核心選取比例統計（50-D）
* <img width="1556" height="804" alt="image" src="https://github.com/user-attachments/assets/77dc26f7-0192-4706-80ea-b855aa94f73a" />

# 研究結論
* 依據上述 50維 優化實驗結果，我們得出以下核心結論：
* 1. 自動化選擇之有效性： 自適應 RBF 核心機制能基於當前訓練點的幾何分佈，動態評估與選擇最優核心類型。在大多數基準函數中，自適應模式均能取得與單一最優核心相當或更好的優化適應度，成功免去手動嘗試的調參痛點。
* 2. 地形與核心偏好： 實驗統計顯示，Cubic 核在平滑且連續性強的地形（如 Ellipsoid 和 Rosenbrock）中展現出高達 65% 以上的主導偏好；而面對高度多峰與含有突變局部極值的地形（如 RHC2 等）時，自適應模式會適度提高 Gaussian 核心的選取比例，以發揮高斯核在局部微調方面的優勢。
* 本實作成功證明了將 Rippa LOOCV 引進 ESA 自適應核心機制的科學合理性，顯著加強了 ESA 優化器在面對各類高維昂貴問題時的泛化與工程自動化能力。

