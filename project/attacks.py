"""Fixed class-capped pilot of masked white-box attacks on a frozen checkpoint.

All metrics refer to exactly the same held-out subset, not the full test set.
Clean input is a candidate: already-misclassified examples remain failures.
Among candidates prefer any misclassification, then maximum cross-entropy.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from wafer_model import CLASS_NAMES, WaferCNN, image_batch, metrics_from_predictions, seed_everything

from artifact_paths import relative_path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def candidate_state(model, x, y):
    logits = model(x)
    ce = F.cross_entropy(logits, y, reduction='none')
    confidence, predicted = logits.softmax(1).max(1)
    return ce, predicted, confidence


def attack_batch(model, x, y, epsilon, kind, steps=20, restarts=5, seed=0):
    clean = x.detach()
    mask = clean > 0
    best_x = clean.clone()
    best_loss, best_pred, best_conf = candidate_state(model, clean, y)

    @torch.no_grad()
    def retain(candidate, loss=None, pred=None, conf=None):
        nonlocal best_x, best_loss, best_pred, best_conf
        if loss is None:
            loss, pred, conf = candidate_state(model, candidate, y)
        wrong, best_wrong = pred.ne(y), best_pred.ne(y)
        choose = (wrong & ~best_wrong) | (wrong.eq(best_wrong) & loss.gt(best_loss))
        best_x[choose] = candidate.detach()[choose]
        best_loss[choose] = loss[choose].detach()
        best_pred[choose] = pred[choose].detach()
        best_conf[choose] = conf[choose].detach()

    if kind == 'fgsm':
        current = clean.clone().requires_grad_(True)
        loss = F.cross_entropy(model(current), y, reduction='sum')
        gradient, = torch.autograd.grad(loss, current)
        candidate = (clean + epsilon * gradient.sign() * mask).clamp(0, 1)
        retain(candidate)
    elif kind == 'pgd':
        generator = torch.Generator(device=clean.device).manual_seed(seed)
        for _ in range(restarts):
            noise = torch.empty_like(clean).uniform_(-epsilon, epsilon, generator=generator)
            current = (clean + noise * mask).clamp(0, 1)
            for _ in range(steps):
                current = current.detach().requires_grad_(True)
                logits = model(current)
                losses = F.cross_entropy(logits, y, reduction='none')
                conf, pred = logits.detach().softmax(1).max(1)
                retain(current, losses.detach(), pred, conf)
                gradient, = torch.autograd.grad(losses.sum(), current)
                with torch.no_grad():
                    delta = (current + (epsilon / 4) * gradient.sign() - clean).clamp(-epsilon, epsilon)
                    current = (clean + delta * mask).clamp(0, 1)
            retain(current)
    else:
        raise ValueError(kind)
    delta = best_x - clean
    assert float(delta.abs().max()) <= epsilon + 1e-6
    assert bool((delta[~mask] == 0).all())
    assert float(best_x.min()) >= 0 and float(best_x.max()) <= 1
    return best_x.detach(), best_pred.detach(), best_conf.detach()


def make_plots(rows, x_clean, x_attacked, labels, predictions, clean_predictions, output):
    table = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for kind, color, marker in [('FGSM', '#386f94', 'o'), ('PGD20-R5', '#bf7044', 's')]:
        selected = table[table.attack == kind]
        for ax, metric in zip(axes, ['accuracy', 'macro_f1']):
            clean_value = table.loc[table.attack == 'Clean', metric].iloc[0]
            ax.plot([0] + selected.epsilon_255.tolist(), [clean_value] + selected[metric].tolist(),
                    color=color, marker=marker, label=kind)
    for ax, metric, title in zip(axes, ['accuracy', 'macro_f1'], ['Accuracy', 'Macro-F1']):
        strong = table[table.attack == 'PGD50-R5']
        ax.scatter(strong.epsilon_255, strong[metric], marker='x', s=75, c='#5e4787', label='PGD50-R5 check')
        ax.set(xlabel='Perturbation bound (numerator / 255)', ylabel=title, ylim=(0, 1), xticks=[0, 2, 4, 8])
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f'Fixed class-capped test subset (n={len(labels)}); single seed')
    fig.tight_layout()
    fig.savefig(output / 'robustness_curves.png', dpi=220)
    plt.close(fig)

    successes = np.flatnonzero((clean_predictions == labels) & (predictions != labels)).tolist()
    chosen = []
    seen = set()
    for idx in successes + list(range(len(labels))):
        if int(labels[idx]) not in seen:
            chosen.append(idx)
            seen.add(int(labels[idx]))
        if len(chosen) == 3:
            break
    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 3 * len(chosen)), squeeze=False)
    for row, idx in enumerate(chosen):
        delta = x_attacked[idx, 0] - x_clean[idx, 0]
        displays = [x_clean[idx, 0], delta, x_attacked[idx, 0]]
        titles = [f'Clean: {CLASS_NAMES[clean_predictions[idx]]}', 'Perturbation (scaled color)',
                  f'PGD50: {CLASS_NAMES[predictions[idx]]}']
        for col, (arr, title) in enumerate(zip(displays, titles)):
            axes[row, col].imshow(arr, cmap='RdBu_r' if col == 1 else 'gray',
                                  vmin=-8/255 if col == 1 else 0, vmax=8/255 if col == 1 else 1,
                                  interpolation='nearest')
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel('True: ' + CLASS_NAMES[labels[idx]], fontsize=10)
    fig.suptitle('Measured attack examples; epsilon = 8/255', fontsize=14)
    fig.tight_layout()
    fig.savefig(output / 'attack_examples.png', dpi=220)
    plt.close(fig)
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'models'/'clean_seed77.pt')
    p.add_argument('--output', type=Path, default=ROOT/'results'/'pilot_evaluation')
    p.add_argument('--per-class', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=64)
    args = p.parse_args()
    if (args.output/'attack_metrics.json').exists():
        raise FileExistsError('Completed attack results already exist; choose a new output directory.')
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(7701)
    device = torch.device('cuda')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    data = ROOT/'data'/'processed'
    for filename, expected in checkpoint['config']['data_sha256'].items():
        if sha256(data/filename) != expected:
            raise RuntimeError('Dataset differs from checkpoint: ' + filename)
    model = WaferCNN().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    images = np.load(data/'images.npy', mmap_mode='r')
    all_labels = np.load(data/'labels.npy')
    with np.load(data/'splits.npz') as f:
        test = f['test']
    rng = np.random.default_rng(7701)
    selected = np.concatenate([np.sort(rng.choice(test[all_labels[test] == c],
                                size=min(args.per_class, int(np.sum(all_labels[test] == c))), replace=False))
                               for c in range(9)])
    labels = all_labels[selected]
    assert len(set(selected.tolist())) == len(selected) and set(selected).issubset(set(test))
    metadata = pd.read_csv(data/'metadata.csv').iloc[selected].copy()
    metadata.to_csv(args.output/'attack_subset_manifest.csv', index=False, encoding='utf-8-sig')
    configuration = {
        'checkpoint': relative_path(args.checkpoint), 'checkpoint_sha256': sha256(args.checkpoint),
        'selected_epoch': checkpoint['epoch'], 'subset_seed': 7701, 'per_class_cap': args.per_class,
        'subset_counts': {name: int(np.sum(labels == c)) for c, name in enumerate(CLASS_NAMES)},
        'n': len(selected), 'epsilon_255': [2,4,8], 'pgd_steps':20, 'restarts':5,
        'pgd_step_size':'epsilon/4', 'strong_check_steps':50, 'batch_size':args.batch_size,
        'selection':'Misclassification first, then greatest unweighted CE across clean and every visited candidate.',
        'scope':'Fixed class-capped held-out pilot subset; aggregate results are not full-test estimates.',
        'threat':'White-box continuous L-infinity post-preprocessing perturbations inside a fixed clean wafer mask.',
        'defect_to_none_scope':'Observed errors under untargeted attack; not a targeted-to-None worst-case rate.',
        'random_starts':'The 20-step and 50-step epsilon8 checks use identical start seeds per batch/restart.',
    }
    (args.output/'attack_config.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    clean_preds, clean_conf = [], []
    clean_images = []
    for start in range(0, len(selected), args.batch_size):
        idx = selected[start:start+args.batch_size]
        x = image_batch(images, idx, device)
        y = torch.tensor(all_labels[idx], device=device)
        _, pred, conf = candidate_state(model, x, y)
        clean_preds.append(pred.cpu().numpy())
        clean_conf.append(conf.cpu().numpy())
        clean_images.append(x.cpu().numpy())
        if start == 0:
            x0, p0, _ = attack_batch(model, x, y, 0, 'fgsm')
            assert torch.equal(x0, x) and torch.equal(p0, pred), 'epsilon0 must match clean'
    clean_preds, clean_conf = np.concatenate(clean_preds), np.concatenate(clean_conf)
    clean_images = np.concatenate(clean_images)
    clean_metrics = metrics_from_predictions(labels, clean_preds)
    rows = [{'attack':'Clean','epsilon_255':0,**{k:clean_metrics[k] for k in ['n','accuracy','macro_f1','defect_to_none_rate']},'seconds':0.0}]
    detailed = {'Clean':clean_metrics}
    predictions = pd.DataFrame({'cache_index':selected,'true_class_id':labels,'clean_predicted_id':clean_preds,
                                'clean_confidence':clean_conf})
    strongest_images = None
    strongest_preds = None
    pgd20_8_preds = None
    started = time.perf_counter()
    experiments = [(kind,eps,20) for kind in ['fgsm','pgd'] for eps in [2,4,8]] + [('pgd',8,50)]
    for kind, numerator, steps in experiments:
        name = 'FGSM' if kind == 'fgsm' else f'PGD{steps}-R5'
        key = f'{name}_eps{numerator}'
        run_start = time.perf_counter()
        preds, confidence, candidates = [], [], []
        for start in range(0, len(selected), args.batch_size):
            idx = selected[start:start+args.batch_size]
            x = image_batch(images, idx, device)
            y = torch.tensor(all_labels[idx], device=device)
            adv, pred, conf = attack_batch(model,x,y,numerator/255,kind,steps=steps,restarts=5,
                                           seed=7701000+numerator*10000+start)
            preds.append(pred.cpu().numpy())
            confidence.append(conf.cpu().numpy())
            if steps == 50:
                candidates.append(adv.cpu().numpy())
        preds, confidence = np.concatenate(preds), np.concatenate(confidence)
        metrics = metrics_from_predictions(labels,preds)
        assert metrics['accuracy'] <= clean_metrics['accuracy']+1e-12
        detailed[key] = metrics
        row = {'attack':name,'epsilon_255':numerator,
               **{k:metrics[k] for k in ['n','accuracy','macro_f1','defect_to_none_rate']},
               'seconds':time.perf_counter()-run_start}
        rows.append(row)
        predictions[key+'_predicted_id'] = preds
        predictions[key+'_confidence'] = confidence
        pd.DataFrame(rows).to_csv(args.output/'attack_metrics.csv',index=False,encoding='utf-8-sig')
        print(json.dumps(row),flush=True)
        if kind == 'pgd' and numerator == 8 and steps == 20:
            pgd20_8_preds = preds
        if steps == 50:
            strongest_images, strongest_preds = np.concatenate(candidates), preds
    assert np.all((pgd20_8_preds == labels) | (strongest_preds != labels)), 'PGD50 must preserve PGD20 successes with shared starts'
    chosen = make_plots(rows, clean_images, strongest_images, labels, strongest_preds, clean_preds,args.output)
    predictions.to_csv(args.output/'attack_predictions.csv',index=False,encoding='utf-8-sig')
    metadata.iloc[chosen].to_csv(args.output/'plotted_attack_examples.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(args.output/'plotted_attack_arrays.npz', clean=clean_images[chosen],attacked=strongest_images[chosen],labels=labels[chosen])
    report = {'config':configuration,'metrics':detailed,'elapsed_seconds':time.perf_counter()-started,
              'checks':{'epsilon_zero_matches_clean':True,'all_candidates_within_linf_bound':True,
                        'background_unchanged':True,'values_within_zero_one':True,
                        'pgd50_preserves_pgd20_successes':True}}
    (args.output/'attack_metrics.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('Completed attack evaluation against the supplied frozen checkpoint.',flush=True)


if __name__ == '__main__':
    main()
