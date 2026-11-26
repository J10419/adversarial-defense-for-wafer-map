"""Render the saved full-test metrics as a readable presentation page."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'clean': 'Clean training', 'noise': 'Uniform-noise training', 'pgd': 'PGD adversarial training'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT / 'results/experiments/summary.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'demo/results.html')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    if summary.get('scope') != 'full_test' or not summary.get('complete'):
        raise ValueError('A completed full-test summary is required.')
    lookup = {(row['method'], row['condition']): row for row in summary['aggregate']}
    def metric(row, name):
        value = row['metrics'][name]
        return f"{100 * value['mean']:.2f} ± {100 * value['sd']:.2f}"
    def rows(condition):
        output = []
        for method, label in LABELS.items():
            row = lookup[method, condition]
            if row['n'] != 25485 or row['seeds'] != [77, 78, 79]:
                raise ValueError('Unexpected evaluation population.')
            output.append(f'<tr><th scope="row">{label}</th><td>{metric(row, "accuracy")}</td>'
                          f'<td>{metric(row, "macro_f1")}</td><td>{metric(row, "defect_to_none_rate")}</td></tr>')
        return ''.join(output)
    template = '''<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Full-test Results · Wafer Map Defense Lab</title><link rel="stylesheet" href="style.css">
<style>.results{max-width:1220px;margin:auto;padding:36px 32px}.results nav{display:flex;justify-content:space-between;gap:20px;margin-bottom:28px;font-size:16px}.results h1{font-size:36px}.results h2{font-size:24px;margin:0 0 10px}.results p{line-height:1.75;font-size:17px;color:var(--muted)}.result-section{background:white;border:1px solid var(--line);border-radius:14px;padding:25px;margin-top:23px;overflow:auto}.results table{min-width:650px;font-size:18px}.results th{font-size:16px}.results td,.results th{padding:16px 10px}.results tbody tr:last-child{background:#e7f3f2}.results tbody th{color:var(--ink);font-weight:600}.results td{font-variant-numeric:tabular-nums;white-space:nowrap}.results .scope{font-size:16px}.results .limit{border-left:4px solid var(--accent);padding-left:18px}.results footer{font-size:14px}@media(max-width:700px){.results{padding:25px 16px}.results h1{font-size:28px}.result-section{padding:18px}.results p{font-size:16px}}</style></head>
<body><main class="results"><nav><a href="index.html">互動展示</a><span>WAFER MAP DEFENSE LAB</span></nav>
<p class="eyebrow">MEASURED RESULTS · FULL TEST SET</p><h1>對抗訓練，如何改變完整測試集的表現？</h1>
<p>25,485 張測試圖 · 三個訓練種子（77、78、79）· 同一個依 lot 分組的資料切分。<br>以下數值為平均 ± 樣本標準差，單位為百分比；macro-F1 對九個類別等權平均。</p>
<section class="result-section"><h2>PGD50-R5 · ε = 8 / 255</h2><p class="scope">50 步、5 次隨機起點。每個模型接受針對自身產生的白箱攻擊。</p>
<table><thead><tr><th scope="col">訓練方法</th><th scope="col">Accuracy ↑</th><th scope="col">Macro-F1 ↑</th><th scope="col">Defect → None ↓</th></tr></thead><tbody>__ATTACK_ROWS__</tbody></table></section>
<section class="result-section"><h2>乾淨輸入 · 無攻擊</h2><table><thead><tr><th scope="col">訓練方法</th><th scope="col">Accuracy ↑</th><th scope="col">Macro-F1 ↑</th><th scope="col">Defect → None ↓</th></tr></thead><tbody>__CLEAN_ROWS__</tbody></table></section>
<p class="limit">限制：乾淨輸入的 Scratch recall 由 36.86% 降為 9.17%。平均穩健性提升不代表每一類都受益。這裡測量連續數位擾動，未驗證物理晶圓修改或其他資料切分的表現。</p>
<footer>完整評估包含 45 個模型／條件組合；來源：results/experiments/summary.json。Defect → None 以 3,753 張真實缺陷圖為分母。</footer>
</main></body></html>'''
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(template.replace('__ATTACK_ROWS__', rows('pgd50')).replace('__CLEAN_ROWS__', rows('clean')), encoding='utf-8')
    print('Rendered saved full-test metrics.')


if __name__ == '__main__':
    main()
