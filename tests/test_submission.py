import sys
sys.path.insert(0, '/home/colin/mapfreechallenge/third_party/mapfree')
import zipfile, io
from pathlib import Path
from benchmark.utils import load_poses, subsample_poses

z = zipfile.ZipFile('/home/colin/mapfreechallenge/gt.zip')
val_root = Path('/home/colin/Documents/val')

total_wanted, total_missing = 0, 0
for scene_dir in sorted(val_root.iterdir()):
    if not scene_dir.is_dir():
        continue
    sid = scene_dir.name
    gt = subsample_poses(load_poses(open(scene_dir/'poses.txt'), load_confidence=False), 5)
    try:
        with z.open(f'pose_{sid}.txt') as f:
            mine = load_poses(io.TextIOWrapper(f, encoding='utf-8'), load_confidence=True)
    except KeyError:
        print(f"{sid}: NO FILE in zip ({len(gt)} frames wanted)")
        total_wanted += len(gt); total_missing += len(gt)
        continue
    miss = set(gt.keys()) - set(mine.keys())
    total_wanted += len(gt); total_missing += len(miss)
    if miss:
        print(f"{sid}: missing {len(miss)}/{len(gt)}  {sorted(miss)[:5]}...")

print(f"\ntotal missing {total_missing}/{total_wanted} = {total_missing/total_wanted:.1%}")