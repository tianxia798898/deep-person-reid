from __future__ import print_function

import argparse
import csv
import json
import os.path as osp
import re
from collections import defaultdict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat

import torchreid
from torchreid import metrics
from torchreid.utils import load_pretrained_weights


FRAME_RE = re.compile(r'C(?P<cam>\d+)T(?P<track>\d+)F(?P<frame>\d+)')


def parse_frame_name(path):
    name = osp.basename(path)
    match = FRAME_RE.search(name)
    if match is None:
        raise ValueError('Cannot parse MARS frame name: {}'.format(name))
    return {
        'camid': int(match.group('cam')) - 1,
        'track': int(match.group('track')),
        'frame': int(match.group('frame')),
        'name': name,
    }


def proxy_time(path):
    meta = parse_frame_name(path)
    return meta['track'] * 1000 + meta['frame']


def make_tracklet_meta(items):
    rows = []
    for idx, item in enumerate(items):
        img_paths, pid, camid = item[:3]
        first = img_paths[0]
        last = img_paths[-1]
        rows.append({
            'index': idx,
            'pid': int(pid),
            'camid': int(camid),
            'start_time': proxy_time(first),
            'end_time': proxy_time(last),
            'first_image': first,
            'last_image': last,
            'num_frames': len(img_paths),
            'img_paths': img_paths,
        })
    return rows


def estimate_camera_windows(train_items, lower_q=5, upper_q=95, margin=1000):
    by_pid = defaultdict(list)
    for item in train_items:
        img_paths, pid, camid = item[:3]
        by_pid[int(pid)].append({
            'camid': int(camid),
            'start_time': proxy_time(img_paths[0]),
            'end_time': proxy_time(img_paths[-1]),
        })

    deltas = defaultdict(list)
    for tracks in by_pid.values():
        for a in tracks:
            for b in tracks:
                if a['camid'] == b['camid']:
                    continue
                deltas[(a['camid'], b['camid'])].append(
                    abs(b['start_time'] - a['end_time'])
                )

    windows = {}
    for pair, values in deltas.items():
        values = np.asarray(values, dtype=np.float32)
        lo = max(0.0, np.percentile(values, lower_q) - margin)
        hi = np.percentile(values, upper_q) + margin
        windows[pair] = (float(lo), float(hi))
    return windows


def apply_time_constraint(distmat, q_meta, g_meta, windows, penalty):
    adjusted = distmat.copy()
    invalid = np.zeros_like(adjusted, dtype=np.bool_)

    for i, q in enumerate(q_meta):
        for j, g in enumerate(g_meta):
            if q['camid'] == g['camid']:
                continue
            pair = (q['camid'], g['camid'])
            if pair not in windows:
                continue
            lo, hi = windows[pair]
            delta = abs(g['start_time'] - q['end_time'])
            if delta < lo or delta > hi:
                invalid[i, j] = True

    adjusted[invalid] += penalty
    return adjusted, invalid


@torch.no_grad()
def extract_features(loader, model, device, pooling='avg'):
    model.eval()
    features, pids, camids = [], [], []
    for batch_idx, data in enumerate(loader):
        imgs = data['img'].to(device)
        b, s, c, h, w = imgs.shape
        imgs = imgs.view(b * s, c, h, w)
        feats = model(imgs).view(b, s, -1)
        if pooling == 'max':
            feats = torch.max(feats, dim=1)[0]
        else:
            feats = torch.mean(feats, dim=1)
        features.append(feats.cpu())
        pids.extend(data['pid'].numpy().tolist())
        camids.extend(data['camid'].numpy().tolist())
        if (batch_idx + 1) % 50 == 0:
            print('  extracted {} batches'.format(batch_idx + 1))
    return torch.cat(features, dim=0), np.asarray(pids), np.asarray(camids)


def compute_scores(distmat, q_pids, g_pids, q_camids, g_camids):
    cmc, m_ap = metrics.evaluate_rank(
        distmat, q_pids, g_pids, q_camids, g_camids, use_cython=False
    )
    return {
        'Rank-1': float(cmc[0] * 100),
        'Rank-5': float(cmc[4] * 100),
        'Rank-10': float(cmc[9] * 100),
        'mAP': float(m_ap * 100),
    }


def first_valid_match(distmat, q_idx, q_pids, g_pids, q_camids, g_camids):
    q_pid = q_pids[q_idx]
    q_camid = q_camids[q_idx]
    order = np.argsort(distmat[q_idx])
    for g_idx in order:
        remove = (g_pids[g_idx] == q_pid) and (g_camids[g_idx] == q_camid)
        if not remove:
            return int(g_idx)
    return None


def top1_transition_stats(dist_base, dist_time, q_pids, g_pids, q_camids, g_camids):
    stats = {
        'total_queries': 0,
        'base_top1_correct': 0,
        'time_top1_correct': 0,
        'rescued_by_time': 0,
        'hurt_by_time': 0,
        'both_wrong': 0,
    }
    for q_idx in range(len(q_pids)):
        b_idx = first_valid_match(dist_base, q_idx, q_pids, g_pids, q_camids, g_camids)
        t_idx = first_valid_match(dist_time, q_idx, q_pids, g_pids, q_camids, g_camids)
        if b_idx is None or t_idx is None:
            continue

        stats['total_queries'] += 1
        base_ok = g_pids[b_idx] == q_pids[q_idx]
        time_ok = g_pids[t_idx] == q_pids[q_idx]
        if base_ok:
            stats['base_top1_correct'] += 1
        if time_ok:
            stats['time_top1_correct'] += 1
        if (not base_ok) and time_ok:
            stats['rescued_by_time'] += 1
        elif base_ok and (not time_ok):
            stats['hurt_by_time'] += 1
        elif (not base_ok) and (not time_ok):
            stats['both_wrong'] += 1

    failed = stats['total_queries'] - stats['base_top1_correct']
    stats['base_top1_error'] = failed
    stats['rescue_rate_in_base_errors'] = (
        100.0 * stats['rescued_by_time'] / failed if failed else 0.0
    )
    stats['net_top1_gain'] = stats['rescued_by_time'] - stats['hurt_by_time']
    stats['net_top1_gain_rate'] = (
        100.0 * stats['net_top1_gain'] / stats['total_queries']
        if stats['total_queries'] else 0.0
    )
    return stats


def write_top1_transition_stats(path, stats):
    with open(path, 'w') as f:
        f.write('| Metric | Value |\n')
        f.write('|---|---:|\n')
        f.write('| Total queries | {total_queries} |\n'.format(**stats))
        f.write('| ReID-only Top-1 correct | {base_top1_correct} |\n'.format(**stats))
        f.write('| ReID+time Top-1 correct | {time_top1_correct} |\n'.format(**stats))
        f.write('| ReID-only Top-1 errors | {base_top1_error} |\n'.format(**stats))
        f.write('| Errors rescued by time constraint | {rescued_by_time} |\n'.format(**stats))
        f.write('| Correct matches hurt by time constraint | {hurt_by_time} |\n'.format(**stats))
        f.write('| Rescue rate among ReID-only Top-1 errors | {rescue_rate_in_base_errors:.2f}% |\n'.format(**stats))
        f.write('| Net Top-1 gain | {net_top1_gain} |\n'.format(**stats))
        f.write('| Net Top-1 gain rate | {net_top1_gain_rate:.2f}% |\n'.format(**stats))


def build_strip(meta, label, width=112, height=224, n_frames=6):
    img_paths = list(meta['img_paths'])
    step = max(1, len(img_paths) // n_frames)
    picked = img_paths[::step][:n_frames]
    font = ImageFont.load_default()
    label_h = 34
    strip = Image.new('RGB', (width * n_frames, height + label_h), 'white')
    draw = ImageDraw.Draw(strip)
    draw.rectangle([0, 0, strip.width, label_h - 1], fill=(30, 30, 30))
    draw.text((6, 8), label, fill='white', font=font)
    for i, path in enumerate(picked):
        img = Image.open(path).convert('RGB').resize((width, height))
        strip.paste(img, (i * width, label_h))
    return strip


def save_rescue_examples(
    dist_base,
    dist_time,
    q_meta,
    g_meta,
    q_pids,
    g_pids,
    q_camids,
    g_camids,
    output_dir,
    max_examples=6
):
    examples = []
    for q_idx in range(len(q_pids)):
        b_idx = first_valid_match(dist_base, q_idx, q_pids, g_pids, q_camids, g_camids)
        t_idx = first_valid_match(dist_time, q_idx, q_pids, g_pids, q_camids, g_camids)
        if b_idx is None or t_idx is None:
            continue
        base_ok = g_pids[b_idx] == q_pids[q_idx]
        time_ok = g_pids[t_idx] == q_pids[q_idx]
        if (not base_ok) and time_ok:
            examples.append((q_idx, b_idx, t_idx))
            if len(examples) >= max_examples:
                break

    if not examples:
        return []

    font = ImageFont.load_default()
    paths = []
    for ex_no, (q_idx, b_idx, t_idx) in enumerate(examples, start=1):
        q = build_strip(
            q_meta[q_idx],
            'Query pid={} C{}'.format(q_pids[q_idx], q_camids[q_idx] + 1)
        )
        b = build_strip(
            g_meta[b_idx],
            'ReID only WRONG pid={} C{}'.format(g_pids[b_idx], g_camids[b_idx] + 1)
        )
        t = build_strip(
            g_meta[t_idx],
            'ReID + time RIGHT pid={} C{}'.format(g_pids[t_idx], g_camids[t_idx] + 1)
        )
        gap = 12
        canvas = Image.new(
            'RGB',
            (q.width, q.height * 3 + gap * 4 + 24),
            (245, 245, 245)
        )
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (8, 8),
            'Example {}: baseline top-1 fails, time-constrained top-1 succeeds'.format(ex_no),
            fill=(0, 0, 0),
            font=font
        )
        y = gap + 24
        for strip in [q, b, t]:
            canvas.paste(strip, (0, y))
            y += strip.height + gap
        path = osp.join(output_dir, 'rescue_example_{}.jpg'.format(ex_no))
        canvas.save(path, quality=92)
        paths.append(path)
    return paths


def remove_stale_examples(output_dir):
    import glob
    import os
    for path in glob.glob(osp.join(output_dir, 'rescue_example_*.jpg')):
        os.remove(path)


def write_scores(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['method', 'Rank-1', 'Rank-5', 'Rank-10', 'mAP'])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='reid-data')
    parser.add_argument('--output-dir', default='outputs/mars_time_constraint')
    parser.add_argument('--model', default='osnet_x0_25')
    parser.add_argument('--weights', default='', help='path to ReID pretrained weights')
    parser.add_argument('--seq-len', type=int, default=4)
    parser.add_argument('--batch-size-test', type=int, default=32)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--penalty', type=float, default=1000.0)
    parser.add_argument('--max-examples', type=int, default=6)
    args = parser.parse_args()

    output_dir = args.output_dir
    if not osp.isdir(output_dir):
        import os
        os.makedirs(output_dir)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    datamanager = torchreid.data.VideoDataManager(
        root=args.root,
        sources='mars',
        targets='mars',
        height=256,
        width=128,
        batch_size_train=2,
        batch_size_test=args.batch_size_test,
        workers=args.workers,
        seq_len=args.seq_len,
        sample_method='evenly',
        use_gpu=(device.type == 'cuda')
    )

    model = torchreid.models.build_model(
        args.model,
        num_classes=datamanager.num_train_pids,
        pretrained=True,
        loss='softmax'
    )
    if args.weights:
        load_pretrained_weights(model, args.weights)
    model = model.to(device)

    query_loader = datamanager.test_loader['mars']['query']
    gallery_loader = datamanager.test_loader['mars']['gallery']
    q_items = datamanager.test_dataset['mars']['query']
    g_items = datamanager.test_dataset['mars']['gallery']
    train_items = datamanager.train_loader.dataset.train
    q_meta = make_tracklet_meta(q_items)
    g_meta = make_tracklet_meta(g_items)

    weight_tag = osp.splitext(osp.basename(args.weights))[0] if args.weights else 'imagenet'
    feature_cache = osp.join(output_dir, 'features_{}_{}_seq{}.pt'.format(args.model, weight_tag, args.seq_len))
    if osp.exists(feature_cache):
        cache = torch.load(feature_cache, map_location='cpu')
        qf = cache['qf']
        gf = cache['gf']
        q_pids = cache['q_pids']
        g_pids = cache['g_pids']
        q_camids = cache['q_camids']
        g_camids = cache['g_camids']
        print('Loaded cached features:', feature_cache)
    else:
        print('Extracting query features ...')
        qf, q_pids, q_camids = extract_features(query_loader, model, device)
        print('Extracting gallery features ...')
        gf, g_pids, g_camids = extract_features(gallery_loader, model, device)
        torch.save({
            'qf': qf,
            'gf': gf,
            'q_pids': q_pids,
            'g_pids': g_pids,
            'q_camids': q_camids,
            'g_camids': g_camids,
        }, feature_cache)
        print('Saved feature cache:', feature_cache)

    print('Computing ReID-only distance matrix ...')
    dist_base = metrics.compute_distance_matrix(qf, gf, metric='euclidean').numpy()
    print('Estimating camera-pair time windows from MARS train split ...')
    windows = estimate_camera_windows(train_items)
    with open(osp.join(output_dir, 'camera_time_windows.json'), 'w') as f:
        json.dump({'{}->{}'.format(k[0] + 1, k[1] + 1): v for k, v in windows.items()}, f, indent=2)

    dist_time, invalid = apply_time_constraint(
        dist_base, q_meta, g_meta, windows, penalty=args.penalty
    )
    print('Invalid pairs penalized: {} / {}'.format(int(invalid.sum()), invalid.size))

    base_scores = compute_scores(dist_base, q_pids, g_pids, q_camids, g_camids)
    time_scores = compute_scores(dist_time, q_pids, g_pids, q_camids, g_camids)
    transition_stats = top1_transition_stats(
        dist_base, dist_time, q_pids, g_pids, q_camids, g_camids
    )

    rows = [
        {'method': 'ReID only', **base_scores},
        {'method': 'ReID + proxy-time constraint', **time_scores},
    ]
    write_scores(osp.join(output_dir, 'metrics.csv'), rows)
    with open(osp.join(output_dir, 'metrics.md'), 'w') as f:
        f.write('| Method | Rank-1 | Rank-5 | Rank-10 | mAP |\n')
        f.write('|---|---:|---:|---:|---:|\n')
        for row in rows:
            f.write('| {method} | {Rank-1:.2f}% | {Rank-5:.2f}% | {Rank-10:.2f}% | {mAP:.2f}% |\n'.format(**row))
    write_top1_transition_stats(
        osp.join(output_dir, 'top1_transition_stats.md'),
        transition_stats
    )

    remove_stale_examples(output_dir)
    example_paths = save_rescue_examples(
        dist_base,
        dist_time,
        q_meta,
        g_meta,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        output_dir,
        max_examples=args.max_examples
    )
    with open(osp.join(output_dir, 'rescue_examples.txt'), 'w') as f:
        for path in example_paths:
            f.write(path + '\n')

    print('\nMetrics')
    for row in rows:
        print('{method}: Rank-1={Rank-1:.2f}% Rank-5={Rank-5:.2f}% Rank-10={Rank-10:.2f}% mAP={mAP:.2f}%'.format(**row))
    print('Top-1 transition stats:')
    for key in sorted(transition_stats):
        print('  {}: {}'.format(key, transition_stats[key]))
    print('Wrote outputs to:', output_dir)
    print('Rescue examples:', len(example_paths))


if __name__ == '__main__':
    main()
