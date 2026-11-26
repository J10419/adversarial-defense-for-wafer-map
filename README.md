# Adversarial Defense for Wafer Map

研究晶圓缺陷分類在數位對抗擾動下的穩健性，比較一般訓練、雜訊增強與 PGD 對抗訓練。專案包含可操作的展示、九個模型、完整測試集的逐筆預測，以及資料處理、訓練與評估程式。

完整研究報告：[PDF](reports/project_report.pdf)。報告區分初步子集分析與完整測試集評估，並保留各類別的限制與失敗分析。

## 快速開始

需要 **Python 3.8 以上與瀏覽器**。展示直接讀取已保存的實測結果，不需要安裝套件、下載原始資料、使用 GPU 或重新訓練。

1. 在本頁按 **Code → Download ZIP**，將 ZIP 完整解壓縮。已有 Git 的使用者也可以執行：

   ```sh
   git clone https://github.com/J10419/adversarial-defense-for-wafer-map.git
   cd adversarial-defense-for-wafer-map
   ```

2. 在包含這份 README 的資料夾開啟終端機，啟動 Python 內建的 HTTP server：

   ```sh
   python -m http.server 8765 --bind 127.0.0.1 --directory demo
   ```

   Windows 若找不到 `python`，改用 `py -3`；macOS／Linux 通常改用 `python3`。若尚未安裝 Python，先安裝 Python 3 再重試。

3. 在瀏覽器開啟：

   - 互動展示：<http://127.0.0.1:8765/>
   - 完整測試結果：<http://127.0.0.1:8765/results.html>

   保持終端機開啟；結束時按 `Ctrl+C`。若 8765 已被占用，把指令與網址中的埠號一起改成 8766。請透過上述網址開啟，直接雙擊 HTML 可能使瀏覽器阻擋資料載入。

互動介面可切換晶圓樣本、訓練模型、攻擊方式與擾動上限，比較原始影像、擾動及分類結果。介面使用固定抽樣的 18 張圖、三個 seed 77 模型與 378 筆保存結果；每個模型的攻擊都針對自身生成。完整測試集的統計另列於結果頁。

## 實驗結果

WM-811K 清理後共有 169,680 筆，以生產批次（lot）分組切分為訓練 118,965、驗證 25,230、測試 25,485 筆。九個類別包含八種缺陷圖樣與 None；None 表示未標註系統性缺陷圖樣，不代表所有晶粒都良好。

下表使用相同的完整測試集，列出三個訓練種子 77、78、79 的平均 ± 樣本標準差，單位為百分比。PGD50-R5 使用 50 步、5 次隨機起點、L∞ 上限 8/255。

| 方法 | 乾淨 Accuracy | 乾淨 macro-F1 | PGD50 Accuracy | PGD50 macro-F1 |
|---|---:|---:|---:|---:|
| Clean training | 95.64 ± 0.23 | 76.83 ± 0.90 | 69.70 ± 12.18 | 42.55 ± 3.05 |
| Uniform-noise training | 95.76 ± 0.09 | 76.91 ± 1.25 | 72.50 ± 11.06 | 44.18 ± 2.25 |
| PGD adversarial training | 95.62 ± 0.16 | 74.00 ± 1.37 | 92.51 ± 0.57 | 62.73 ± 0.84 |

另外已完成 FGSM、PGD20 與 targeted-to-None 評估，共 45 個模型／測試條件組合。targeted-to-None 評估使用測試集中的 3,753 張真實缺陷圖，分母不同於上表。

對抗訓練改善了這個設定下的整體穩健性，但不是所有類別都受益：乾淨輸入的 Scratch recall 從 36.86% 降到 9.17%。類別不平衡也使 Accuracy 無法單獨代表少數類別的表現。三個種子的標準差只描述固定資料切分下的訓練變動，不是跨批次或跨切分的信賴區間。

## 方法與攻擊範圍

- 輸入：64×64 晶圓圖，原始 0/1/2 正規化為 0/0.5/1。
- 模型：三層 32/64/128 通道卷積、ReLU、max pooling、global average pooling 與九類線性輸出，共 93,833 個參數。
- 訓練：相同資料與模型，比較 clean、固定晶圓區域內的 uniform noise，以及 PGD 對抗訓練。對抗訓練使用 7 步 PGD，損失為各半的 clean／adversarial weighted cross entropy。
- 評估：模型特定的白箱攻擊；擾動只作用於固定晶圓區域，並裁切到合法輸入範圍。這是連續數位擾動，不代表物理晶圓遭到修改或實際晶粒狀態被翻轉。
- 展示的信心分數是未校準的 softmax 輸出；中央色圖放大擾動，右側才是實際送入模型的影像。

## 專案結構

```text
demo/                  互動介面、影像、保存的模型輸出與結果頁
reports/               完整研究報告與來源核對紀錄
project/               資料處理、訓練、攻擊與驗證程式
models/                九個模型與權重一致性紀錄
results/experiments/   45 組逐筆預測、指標與評估協定
data/                  資料來源與固定切分；不含大型原始資料及影像快取
requirements.txt       資料處理、訓練與評估所需套件
```

## 重現與驗證

展示所需的 Python HTTP server 與研究用環境彼此獨立。研究程式使用 Python 3.12；資料下載、GPU 推論及重訓需要額外的套件、空間與時間。

公開模型移除了原執行環境的路徑資訊；模型張量保持一致。原始執行雜湊與可攜版本的雜湊分開記錄，詳見模型及結果目錄中的來源紀錄。重新整理中繼資料不代表重新執行實驗。

Windows PowerShell 範例（請以 Python 3.12 建立研究環境）：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe project/verify_models.py
.\.venv\Scripts\python.exe project/verify_results.py --results results/experiments
```

macOS／Linux 可用 `python3.12 -m venv .venv`，後續將 `.\.venv\Scripts\python.exe` 換成 `.venv/bin/python`。上述驗證會核對公開模型、保存的預測與中繼資料，不會重新執行 GPU 攻擊，也不會把未附的影像快取視為已檢查。

需要重現資料與推論時，先取得資料並重建影像快取：

```powershell
.\.venv\Scripts\python.exe project/prepare_data.py
.\.venv\Scripts\python.exe project/prepare_splits.py
.\.venv\Scripts\python.exe project/verify_results.py --results results/experiments --require-images
```

CUDA 評估需要 NVIDIA GPU 與支援該 GPU 的 PyTorch 安裝；已保存實驗使用 PyTorch 2.10.0 / CUDA 12.8。請先依自己的硬體準備 CUDA 版 PyTorch，再執行完整評估。輸出到新目錄以保留既有結果：

```powershell
.\.venv\Scripts\python.exe project/experiments.py --output results/reproduction --phase all
```

重新訓練與重建展示的範例：

```powershell
.\.venv\Scripts\python.exe project/train.py --mode clean --seed 77 --epochs 10 --run-dir results/retraining/clean_seed77
.\.venv\Scripts\python.exe project/build_demo.py --output results/demo-rebuild --device cpu
.\.venv\Scripts\python.exe project/build_results_view.py
```

訓練的 `--mode` 可選 `clean`、`noise`、`pgd`；每次實驗使用新的輸出路徑。`build_results_view.py` 只從保存的彙整數據更新展示結果頁，不會執行模型推論。

## 資料來源

[WM-811K dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)。資料來源、下載檔雜湊與固定切分一併保存；原始資料需依來源平台的使用條件取得。

M.-J. Wu, J.-S. R. Jang, and J.-L. Chen. “Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets.” IEEE Transactions on Semiconductor Manufacturing, 28(1):1–12, 2015. DOI: [10.1109/TSM.2014.2364237](https://doi.org/10.1109/TSM.2014.2364237).
