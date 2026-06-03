from __future__ import print_function

import argparse
import csv
import os.path as osp
import re

from scipy.io import loadmat


FRAME_PATTERN = re.compile(r'C(?P<camid>\d+)T(?P<track>\d+)F(?P<frame>\d+)')


def read_names(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def parse_name(name):
    match = FRAME_PATTERN.search(name)
    if match is None:
        return {'track': '', 'frame': ''}
    return {
        'track': int(match.group('track')),
        'frame': int(match.group('frame')),
    }


def build_rows(names, track_info, subset):
    rows = []
    for idx, item in enumerate(track_info):
        start_index, end_index, pid, camid = [int(x) for x in item]
        first_name = names[start_index - 1]
        last_name = names[end_index - 1]
        first_meta = parse_name(first_name)
        last_meta = parse_name(last_name)
        rows.append({
            'subset': subset,
            'tracklet_id': idx,
            'pid': pid,
            'camid': camid - 1,
            'start_index': start_index,
            'end_index': end_index,
            'num_frames': end_index - start_index + 1,
            'mars_track': first_meta['track'],
            'start_frame': first_meta['frame'],
            'end_frame': last_meta['frame'],
            'first_image': first_name,
            'last_image': last_name,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='reid-data/mars')
    parser.add_argument('--output', default='reid-data/mars/mars_tracklets.csv')
    args = parser.parse_args()

    info_dir = osp.join(args.root, 'info')
    test_names = read_names(osp.join(info_dir, 'test_name.txt'))
    track_test = loadmat(osp.join(info_dir, 'tracks_test_info.mat'))[
        'track_test_info'
    ]
    query_idx = loadmat(osp.join(info_dir, 'query_IDX.mat'))['query_IDX'].squeeze()
    query_idx -= 1
    query_set = set(int(x) for x in query_idx)

    query_rows = build_rows(test_names, track_test[query_idx, :], 'query')
    gallery_idx = [i for i in range(track_test.shape[0]) if i not in query_set]
    gallery_rows = build_rows(test_names, track_test[gallery_idx, :], 'gallery')
    rows = query_rows + gallery_rows

    fieldnames = [
        'subset', 'tracklet_id', 'pid', 'camid', 'start_index', 'end_index',
        'num_frames', 'mars_track', 'start_frame', 'end_frame', 'first_image',
        'last_image'
    ]
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print('Wrote {} rows to {}'.format(len(rows), args.output))


if __name__ == '__main__':
    main()
