"""Build an offline gallery from real held-out wafers and frozen checkpoints.

The browser reads saved model outputs; it never calls an API or invents scores.
Every attack is generated separately against its selected model. Cache reuse
requires matching data, checkpoint, source-code and protocol hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from PIL import Image
import torch

from attacks import attack_batch, sha256
from wafer_model import CLASS_NAMES, WaferCNN, image_batch, seed_everything

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def relative(path, output):
    return Path(path).relative_to(output).as_posix()


def png_gray(values, path):
    Image.fromarray(np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)).save(path)


def png_delta(delta, epsilon, path):
    # A signed, fixed-scale map: zero white, negative blue, positive red.
    scaled = np.clip(delta / epsilon, -1, 1) if epsilon else np.zeros_like(delta)
    rgb = np.ones((*delta.shape, 3), dtype=np.float32)
    strength = np.abs(scaled)
    rgb[scaled < 0, 0] -= strength[scaled < 0] * .83
    rgb[scaled < 0, 1] -= strength[scaled < 0] * .60
    rgb[scaled >= 0, 1] -= strength[scaled >= 0] * .72
    rgb[scaled >= 0, 2] -= strength[scaled >= 0] * .75
    Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).save(path)


def default_models(seeds):
    result = []
    for seed in seeds:
        for mode, label in [('clean', 'Clean training'), ('noise', 'Uniform noise'), ('pgd', 'PGD adversarial training')]:
            checkpoint = ROOT / 'models' / f'{mode}_seed{seed}.pt'
            result.append({'id': f'{mode}_seed{seed}', 'label': label, 'seed': seed,
                           'checkpoint': str(checkpoint)})
    return result


def load_extra_models(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, list):
        raise ValueError('--models-json must contain a list of id, label, seed, checkpoint objects.')
    return value


def build(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    asset_dir = output / 'assets'
    cache_dir = output / 'cache'
    asset_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    for name in ['index.html', 'style.css', 'app.js']:
        source = ROOT / 'demo' / name
        if source.is_file() and source.resolve() != (output / name).resolve():
            shutil.copy2(source, output / name)

    seed_everything(args.selection_seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; use --device cpu.')
    data = ROOT / 'data' / 'processed'
    data_hashes = {name: sha256(data / name) for name in ['images.npy', 'labels.npy', 'splits.npz', 'metadata.csv']}
    images = np.load(data / 'images.npy', mmap_mode='r')
    all_labels = np.load(data / 'labels.npy', mmap_mode='r')
    with np.load(data / 'splits.npz') as split_file:
        test = np.array(split_file['test'], dtype=np.int64)
    metadata = pd.read_csv(data / 'metadata.csv', keep_default_na=False).set_index('cache_index')
    generator = np.random.default_rng(args.selection_seed)
    selected = []
    for class_id in range(len(CLASS_NAMES)):
        eligible = np.sort(test[all_labels[test] == class_id])
        if not len(eligible):
            raise ValueError(f'Test split has no examples of {CLASS_NAMES[class_id]}')
        selected.extend(int(i) for i in generator.choice(eligible, min(args.samples_per_class, len(eligible)), replace=False))
    if args.smoke:
        selected = selected[:1] + selected[-1:]
    if len(set(selected)) != len(selected) or not set(selected).issubset(set(test.tolist())):
        raise AssertionError('Demo samples must be unique held-out test indices.')

    samples = []
    for idx in selected:
        row = metadata.loc[idx]
        clean_path = asset_dir / f'clean_{data_hashes["images.npy"][:12]}_{idx}.png'
        png_gray(np.asarray(images[idx], dtype=np.float32) / 2, clean_path)
        samples.append({'id': str(idx), 'cache_index': idx, 'class_id': int(all_labels[idx]),
                        'label': CLASS_NAMES[int(all_labels[idx])], 'lot': str(row['lot']),
                        'original_row': int(row['original_row']), 'split': str(row['split']),
                        'raw_hash': str(row['raw_hash']), 'resized_hash': str(row['resized_hash']),
                        'clean_image': relative(clean_path, output)})
    epsilons = [8] if args.smoke else sorted(set(args.epsilon))
    configurations = [{'id': 'clean', 'label': 'Clean', 'kind': 'clean', 'epsilon_255': 0, 'steps': 0, 'restarts': 0}]
    for epsilon in epsilons:
        configurations += [
            {'id': f'fgsm_{epsilon}', 'label': 'FGSM', 'kind': 'fgsm', 'epsilon_255': epsilon, 'steps': 1, 'restarts': 1},
            {'id': f'pgd20_{epsilon}', 'label': 'PGD20-R5', 'kind': 'pgd', 'epsilon_255': epsilon, 'steps': 20, 'restarts': 5},
        ]
    specs = default_models(args.seeds)
    if args.models_json:
        specs += load_extra_models(args.models_json)
    if len({spec['id'] for spec in specs}) != len(specs):
        raise ValueError('Each model must have a unique id.')
    source_hashes = {name: sha256(ROOT / 'project' / name) for name in ['build_demo.py', 'wafer_model.py', 'attacks.py']}
    model_records, records, cache_events = [], [], []
    start = time.monotonic()
    for spec in specs:
        if not all(c.isalnum() or c in '_-' for c in spec['id']):
            raise ValueError('Model id may contain only letters, numbers, underscores and hyphens.')
        checkpoint_path = Path(spec['checkpoint'])
        if not checkpoint_path.is_absolute():
            checkpoint_path = ROOT / checkpoint_path
        checkpoint_hash = sha256(checkpoint_path)
        model_record = {'id': spec['id'], 'label': spec['label'], 'seed': int(spec['seed']),
                        'checkpoint': checkpoint_path.relative_to(ROOT).as_posix() if checkpoint_path.is_relative_to(ROOT) else checkpoint_path.name,
                        'checkpoint_sha256': checkpoint_hash}
        protocol = {'schema': SCHEMA, 'model': model_record, 'data_sha256': data_hashes,
                    'source_sha256': source_hashes, 'samples': selected, 'selection_seed': args.selection_seed,
                    'configurations': configurations, 'device': str(device), 'torch': torch.__version__,
                    'numpy': np.__version__, 'attack_seed': 770500,
                    'candidate_policy': 'Prefer a misclassification, then largest CE; retain clean input as a candidate.',
                    'mask': 'Fixed clean input > 0', 'clip': [0, 1], 'pgd_step_size': 'epsilon/4'}
        fingerprint = digest_json(protocol)
        cache_path = cache_dir / f'{spec["id"]}_{fingerprint[:16]}.json'
        cached = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else None
        if cached and cached.get('fingerprint') == fingerprint and all((output / name).is_file() and sha256(output / name) == expected for name, expected in cached.get('artifact_sha256', {}).items()) and cached.get('artifact_sha256'):
            records.extend(cached['records'])
            model_records.append(cached['model'])
            cache_events.append({'model': spec['id'], 'status': 'reused', 'fingerprint': fingerprint})
            print(f'{spec["id"]}: reused verified cache', flush=True)
            continue

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if checkpoint['config']['data_sha256'] != data_hashes:
            raise ValueError(f'{spec["id"]}: data does not match the training checkpoint.')
        if int(checkpoint['config']['seed']) != int(spec['seed']):
            raise ValueError(f'{spec["id"]}: model seed does not match the checkpoint.')
        model = WaferCNN().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model_record['selected_epoch'] = int(checkpoint['epoch'])
        x = image_batch(images, selected, device)
        y = torch.tensor(np.asarray(all_labels[selected]).copy(), dtype=torch.long, device=device)
        original = x.detach().cpu().numpy()[:, 0]
        saved_arrays = {'cache_indices': np.array(selected), 'labels': y.cpu().numpy(), 'clean': original}
        model_output, artifact_paths = [], []
        for config in configurations:
            if config['kind'] == 'clean':
                attacked = x.clone()
            else:
                attacked, _, _ = attack_batch(model, x, y, config['epsilon_255'] / 255,
                                              config['kind'], config['steps'], config['restarts'], seed=770500)
            with torch.no_grad():
                probabilities = model(attacked).softmax(1).cpu().numpy()
            attacked_np = attacked.cpu().numpy()[:, 0]
            saved_arrays[config['id']] = attacked_np
            saved_arrays[f'{config["id"]}_probabilities'] = probabilities
            for offset, sample in enumerate(samples):
                delta = attacked_np[offset] - original[offset]
                epsilon = config['epsilon_255'] / 255
                if not np.isfinite(probabilities[offset]).all() or abs(float(probabilities[offset].sum()) - 1) > 1e-5:
                    raise AssertionError('Invalid model probabilities.')
                if np.abs(delta).max() > epsilon + 1e-6 or np.any(delta[original[offset] == 0] != 0):
                    raise AssertionError('Attack violates its bound or the clean wafer mask.')
                prefix = f'{spec["id"]}_{fingerprint[:12]}_{sample["id"]}_{config["id"]}'
                attack_path, delta_path = asset_dir / f'{prefix}.png', asset_dir / f'{prefix}_delta.png'
                png_gray(attacked_np[offset], attack_path)
                png_delta(delta, epsilon, delta_path)
                artifact_paths += [attack_path, delta_path]
                predicted = int(probabilities[offset].argmax())
                model_output.append({'model_id': spec['id'], 'sample_id': sample['id'], 'config_id': config['id'],
                                     'predicted_class': predicted, 'prediction': CLASS_NAMES[predicted],
                                     'confidence': float(probabilities[offset, predicted]),
                                     'probabilities': probabilities[offset].astype(float).tolist(),
                                     'correct': predicted == sample['class_id'],
                                     'linf': float(np.abs(delta).max()), 'changed_pixels': int(np.count_nonzero(delta)),
                                     'image': relative(attack_path, output), 'delta_image': relative(delta_path, output)})
            print(f'{spec["id"]}: {config["id"]}, {len(samples)} real samples', flush=True)
        array_path = cache_dir / f'{spec["id"]}_{fingerprint[:16]}.npz'
        np.savez_compressed(array_path, **saved_arrays)
        artifact_paths.append(array_path)
        model_record['arrays'] = relative(array_path, output)
        model_record['provenance'] = relative(cache_path, output)
        write_json(cache_path, {'fingerprint': fingerprint, 'protocol': protocol, 'model': model_record,
                                'records': model_output,
                                'artifact_sha256': {relative(path, output): sha256(path) for path in artifact_paths}})
        model_records.append(model_record)
        records.extend(model_output)
        cache_events.append({'model': spec['id'], 'status': 'generated', 'fingerprint': fingerprint})
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    payload = {'schema': SCHEMA, 'title': 'Wafer Map Defense Lab', 'built_at_utc': datetime.now(timezone.utc).isoformat(),
               'classes': CLASS_NAMES, 'samples': samples, 'models': model_records, 'configurations': configurations,
               'records': records, 'data_sha256': data_hashes, 'source_sha256': source_hashes,
               'selection': {'rule': 'Uniform random choice within each true class of the held-out test split, without replacement; no outcome-based filtering.',
                             'seed': args.selection_seed, 'requested_per_class': args.samples_per_class, 'smoke': args.smoke},
               'scope': 'A small, fixed demonstration gallery; do not interpret its accuracy as test-set performance.',
               'threat_model': 'Continuous digital perturbations in normalized input space, masked to the original wafer area; not physical die flips.',
               'none_meaning': 'None means no labeled systematic defect pattern. It does not mean all dies passed.',
               'cache_events': cache_events, 'build_seconds': round(time.monotonic() - start, 3)}
    write_json(output / 'dataset.json', payload)
    check = {'sample_count': len(samples), 'model_count': len(model_records), 'record_count': len(records),
             'expected_record_count': len(samples) * len(model_records) * len(configurations),
             'all_test_samples': all(sample['split'] == 'test' for sample in samples),
             'all_images_exist': all((output / record[key]).is_file() for record in records for key in ['image', 'delta_image']),
             'data_sha256': data_hashes, 'cache_events': cache_events}
    if check['record_count'] != check['expected_record_count'] or not check['all_test_samples'] or not check['all_images_exist']:
        raise AssertionError('Demo verification failed.')
    write_json(output / 'build_checks.json', check)
    print(json.dumps({'output': str(output), **check}, ensure_ascii=False), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'demo')
    parser.add_argument('--smoke', action='store_true', help='Only two test wafers and epsilon 8/255.')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--samples-per-class', type=int, default=2)
    parser.add_argument('--selection-seed', type=int, default=7751)
    parser.add_argument('--seeds', nargs='+', type=int, default=[77])
    parser.add_argument('--epsilon', nargs='+', type=int, default=[2, 4, 8])
    parser.add_argument('--models-json', type=Path, help='Optional extra frozen WaferCNN model specifications.')
    args = parser.parse_args()
    if args.samples_per_class < 1 or args.cpu_threads < 1 or any(e <= 0 or e > 255 for e in args.epsilon):
        parser.error('samples-per-class and cpu-threads must be positive; epsilon must be in 1..255.')
    return args


if __name__ == '__main__':
    build(parse_args())
