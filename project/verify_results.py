"""Independently recalculate metrics from saved per-map predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['Center','Donut','Edge-Loc','Edge-Ring','Loc','Random','Scratch','Near-full','None']


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def receipt(folder, value):
    temporary = folder/'verification.json.tmp'
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(temporary, folder/'verification.json')


def calculate(y, p):
    cm = np.zeros((9, 9), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    tp = np.diag(cm)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2*tp, denom, out=np.zeros(9), where=denom != 0)
    supports = cm.sum(1)
    recalls = np.divide(tp, supports, out=np.zeros(9), where=supports != 0)
    defects = y != 8
    count = int(np.sum(defects & (p == 8)))
    return dict(n=len(y), accuracy=float(tp.sum()/len(y)), macro_f1=float(f1.mean()),
                confusion_matrix=cm.tolist(), per_class_support=dict(zip(CLASSES, supports.tolist())),
                per_class_recall=dict(zip(CLASSES, recalls.tolist())),
                true_defect_count=int(defects.sum()), defect_to_none_count=count,
                defect_to_none_rate=float(count/defects.sum()) if defects.any() else None)


def compare(actual, expected, where):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key not in actual:
                raise AssertionError(f'{where}: missing {key}')
            compare(actual[key], value, where + '.' + key)
    elif isinstance(expected, list):
        assert actual == expected, where
    elif expected is None:
        assert actual is None, where
    elif isinstance(expected, str):
        assert actual == expected, where
    else:
        assert np.isclose(actual, expected, atol=1e-10, rtol=1e-10), (where, actual, expected)


def verify(folder, require_images=False):
    folder = Path(folder)
    receipt(folder, {'passed':False, 'status':'verification_in_progress'})
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8-sig'))
    protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8-sig'))
    assert object_hash(protocol['definition']) == protocol['protocol_sha256'] == summary['protocol_sha256']
    missing_optional_data = []
    for name, expected in protocol['definition']['data_sha256'].items():
        path = ROOT/'data/processed'/name
        if name == 'images.npy' and not path.exists() and not require_images:
            missing_optional_data.append(name)
            continue
        assert file_hash(path) == expected, 'Data hash changed: ' + name
    for name, expected in protocol['definition']['code_sha256'].items():
        assert file_hash(ROOT/name) == expected, 'Experiment source changed: ' + name
    for record in protocol['definition']['runs']:
        assert file_hash(ROOT/record['checkpoint']) == record['checkpoint_sha256'], 'Checkpoint changed'
    labels = np.load(ROOT/'data/processed/labels.npy')
    with np.load(ROOT/'data/processed/splits.npz') as splits:
        test = splits['test'].astype(int)
    records = {}
    hashes = {}
    for run in summary['runs']:
        path = (folder/run['predictions_path']).resolve()
        assert path.is_relative_to(folder.resolve()), 'Prediction path escapes output'
        metrics_path = (folder/run['metrics_path']).resolve()
        assert metrics_path.is_relative_to(folder.resolve())
        saved = json.loads(metrics_path.read_text(encoding='utf-8-sig'))
        assert saved['protocol_sha256'] == protocol['protocol_sha256']
        assert saved['config_sha256'] == object_hash(saved['config'])
        assert saved['predictions_sha256'] == file_hash(path), 'Predictions hash changed'
        assert saved['metrics'] == run['metrics'], 'Summary and per-run metrics differ'
        frame = pd.read_csv(path)
        ids = frame.cache_index.to_numpy(dtype=int)
        y = frame.true_class_id.to_numpy(dtype=int)
        pred = frame.predicted_id.to_numpy(dtype=int)
        assert len(ids) > 0 and len(set(ids)) == len(ids), 'Empty/duplicate prediction rows'
        assert np.isin(ids, test).all(), 'Prediction contains non-test rows'
        assert np.array_equal(y, labels[ids]), 'Ground truth differs from frozen data'
        assert ((pred >= 0) & (pred < 9)).all(), 'Invalid class id'
        assert np.isfinite(frame.confidence).all() and frame.confidence.between(0, 1).all()
        if run['condition'] == 'targeted_none':
            assert (y != 8).all()
            expected_ids = test[labels[test] != 8]
        else:
            expected_ids = test
        if summary['scope'] == 'full_test':
            assert np.array_equal(np.sort(ids), np.sort(expected_ids)), 'Incorrect full-test population'
        else:
            assert summary['scope'] == 'smoke'
        computed = calculate(y, pred)
        if run['condition'] == 'targeted_none':
            initial = frame.clean_predicted_id.to_numpy(dtype=int)
            correct = initial == y
            success = pred == 8
            baseline = initial == 8
            computed.update(target_class='None', target_class_id=8,
                target_success_count=int(success.sum()), target_success_denominator=len(y),
                target_success_rate=float(success.mean()),
                clean_correct_defect_success_count=int((success & correct).sum()),
                clean_correct_defect_denominator=int(correct.sum()),
                clean_correct_defect_success_rate=float(success[correct].mean()) if correct.any() else None,
                baseline_clean_defect_to_none_count=int(baseline.sum()),
                baseline_clean_defect_to_none_rate=float(baseline.mean()),
                new_target_success_count=int((success & ~baseline).sum()))
        compare(run['metrics'], computed, str(path))
        assert run['n'] == len(ids)
        key = (run['method'], int(run['seed']), run['condition'])
        assert key not in records, 'Duplicate run key'
        records[key] = (run, frame.set_index('cache_index').sort_index(), computed)
        hashes[run['predictions_path']] = file_hash(path)
    for (method, seed, condition), (run, frame, computed) in records.items():
        if condition == 'clean':
            continue
        clean = records[(method, seed, 'clean')][1].loc[frame.index]
        if 'clean_predicted_id' in frame:
            assert np.array_equal(frame.clean_predicted_id, clean.predicted_id)
        if condition != 'targeted_none':
            assert not ((clean.predicted_id != clean.true_class_id) &
                        (frame.predicted_id == frame.true_class_id)).any(), 'Attack forgot clean errors'
        if condition == 'pgd50':
            short = records[(method, seed, 'pgd20')][1].loc[frame.index]
            assert not ((short.predicted_id != short.true_class_id) &
                        (frame.predicted_id == frame.true_class_id)).any(), 'PGD50 lost PGD20 failures'
        if condition == 'targeted_none':
            assert not ((clean.predicted_id == 8) & (frame.predicted_id != 8)).any(), 'Target attack forgot clean successes'
    for aggregate in summary['aggregate']:
        assert sorted(aggregate['seeds']) == [77, 78, 79]
        selected = [records[(aggregate['method'], seed, aggregate['condition'])][2] for seed in [77,78,79]]
        metric_names = ['accuracy', 'macro_f1', 'defect_to_none_rate']
        if aggregate['condition'] == 'targeted_none':
            metric_names += ['target_success_rate','clean_correct_defect_success_rate','baseline_clean_defect_to_none_rate']
        for metric in metric_names:
            values = [item[metric] for item in selected]
            expected = {'mean':None,'sd':None} if any(v is None for v in values) else {'mean':float(np.mean(values)), 'sd':float(np.std(values, ddof=1))}
            compare(aggregate['metrics'][metric], expected, metric)
        for name in CLASSES:
            values = [item['per_class_recall'][name] for item in selected]
            compare(aggregate['per_class_recall'][name], {'mean':float(np.mean(values)), 'sd':float(np.std(values,ddof=1))}, name)
    for condition in summary['complete_conditions']:
        assert all((method, seed, condition) in records for method in ['clean','noise','pgd'] for seed in [77,78,79])
    result = {'passed':True, 'scope':summary['scope'], 'runs_checked':len(records),
              'summary_sha256':file_hash(folder/'summary.json'), 'protocol_sha256':protocol['protocol_sha256'],
              'aggregate_rows_checked':len(summary['aggregate']), 'predictions_sha256':hashes,
              'missing_optional_data':missing_optional_data, 'image_cache_verified':not missing_optional_data,
              'verification_scope':'Saved predictions, source/model hashes and available data manifests; no inference replay',
              'checks':['frozen test identities','ground truth','independent confusion/F1/recall/D-to-N',
                        'sample SD','paired clean identities','retained attack successes']}
    receipt(folder, result)
    print(json.dumps({k:v for k,v in result.items() if k != 'predictions_sha256'}))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results/experiments')
    parser.add_argument('--require-images', action='store_true', help='Also require the rebuilt large image cache and check its recorded SHA-256.')
    args = parser.parse_args()
    verify(args.results, require_images=args.require_images)
