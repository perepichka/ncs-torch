"""Scan LAFAN1 (original 30 fps BVH) for stand<->crouch transitions, crouch locomotion and jumps.

Times are reported in seconds so they transfer to lafan1-resolved (same captures, re-solved at 60 fps).
Heuristic only: every candidate must be confirmed in the playground clip browser.

Usage:  python scan_lafan1.py <dir with original LAFAN1 *.bvh> [out_dir]
Needs numpy and `bvh.py` + `quat.py` from orangeduck/GenoView/resources (MIT) on PYTHONPATH.
Outputs lafan1_crouch_transitions.csv and lafan1_jump_candidates.csv (committed in docs/playground/data/).
"""
import glob, os, sys
import numpy as np
import bvh, quat

FPS = 30.0
ROOT = sys.argv[1]
files = sorted(glob.glob(os.path.join(ROOT, "*.bvh")))

def load(f):
    d = bvh.load(f)
    rot = quat.unroll(quat.from_euler(np.radians(d['rotations']), order=d['order']))
    _, gp = quat.fk(rot, d['positions'], d['parents'])
    return d['names'], gp / 100.0  # cm -> m

def medfilt(x, k=7):
    p = k // 2; xp = np.pad(x, (p, p), mode='edge')
    return np.median(np.stack([xp[i:i + len(x)] for i in range(k)]), axis=0)

cache = {}
def get(f):
    if f not in cache: cache[f] = load(f)
    return cache[f]

# Per-subject standing references from walk clips
ref = {}
for f in files:
    b = os.path.basename(f)
    if b.startswith('walk'):
        s = b.split('_')[1][:-4]
        names, gp = get(f)
        ref.setdefault(s, {'hip': [], 'head': []})
        ref[s]['hip'].append(np.median(gp[:, names.index('Hips'), 1]))
        ref[s]['head'].append(np.median(gp[:, names.index('Head'), 1]))
ref = {s: (np.mean(v['hip']), np.mean(v['head'])) for s, v in ref.items()}
print("standing refs (hips, head) m:", {s: (round(a, 3), round(b, 3)) for s, (a, b) in ref.items()})

CATS = ('ground', 'obstacles', 'multipleActions', 'jumps', 'aiming', 'fallAndGetUp', 'fightAndSports', 'dance')
trans_rows, crouch_rows, jump_rows = [], [], []
for f in files:
    b = os.path.basename(f)[:-4]
    if not b.startswith(CATS): continue
    s = b.split('_')[1]
    names, gp = get(f)
    T = len(gp)
    hip = gp[:, names.index('Hips')]; head = gp[:, names.index('Head')]
    r = medfilt(hip[:, 1] / ref[s][0])
    headr = medfilt(head[:, 1] / ref[s][1])
    vel = np.zeros(T); vel[1:] = np.linalg.norm(np.diff(hip[:, [0, 2]], axis=0), axis=1) * FPS
    vel = medfilt(vel, 9)
    # state per frame: 2 stand, 1 crouch, 0 low (crawl/prone/sitting), -1 ambiguous
    state = np.full(T, -1)
    state[r > 0.88] = 2
    state[(r > 0.45) & (r < 0.78) & (headr > 0.55)] = 1
    state[headr < 0.45] = 0
    # crouch locomotion time
    cm = (state == 1)
    crouch_rows.append((b, cm.sum() / FPS, (cm & (vel > 0.4)).sum() / FPS, np.percentile(vel[cm], 90) if cm.any() else 0))
    # transitions: last stable stand frame -> first stable crouch frame (or reverse) within 1.5 s, no 'low' in between
    stable = []
    i = 0
    while i < T:
        j = i
        while j + 1 < T and state[j + 1] == state[i]: j += 1
        if state[i] in (1, 2) and (j - i + 1) >= int(0.3 * FPS): stable.append((state[i], i, j))
        i = j + 1
    for (sa, a0, a1), (sb, b0, b1) in zip(stable, stable[1:]):
        if sa != sb and (b0 - a1) <= 1.5 * FPS and not (state[a1:b0] == 0).any():
            kind = 'stand->crouch' if sa == 2 else 'crouch->stand'
            trans_rows.append((b, kind, a1 / FPS, b0 / FPS, (b0 - a1) / FPS, vel[a1:b0 + 1].mean()))
    # jumps: all foot joints above ground, feet moving, short duration
    feet = [names.index(n) for n in ('LeftFoot', 'LeftToe', 'RightFoot', 'RightToe')]
    fy = gp[:, feet, 1]
    ground = np.percentile(gp[:, [names.index('LeftToe'), names.index('RightToe')], 1], 2)
    fv = np.zeros((T, 4)); fv[1:] = np.linalg.norm(np.diff(gp[:, feet], axis=0), axis=2) * FPS
    air = (fy.min(axis=1) > ground + 0.08) & (fv.min(axis=1) > 0.3)
    i = 0
    while i < T:
        if air[i]:
            j = i
            while j + 1 < T and air[j + 1]: j += 1
            dur = (j - i + 1) / FPS
            if 0.15 <= dur <= 1.0:
                pre = max(0, i - 3)
                rise = hip[i:j + 1, 1].max() - hip[pre, 1]
                jump_rows.append((b, i / FPS, (j + 1) / FPS, dur, vel[pre], rise))
            i = j + 1
        else:
            i += 1

print("\n== crouch coverage (seconds): clip, crouch total, crouch moving (>0.4 m/s), p90 crouch speed")
for row in crouch_rows:
    if row[1] > 2: print("  %-28s %6.1f  %6.1f  %.2f m/s" % row)
print("\n== stand<->crouch transition candidates: clip, kind, start s, end s, duration s, mean root speed")
for row in trans_rows: print("  %-28s %-14s %7.2f %7.2f %5.2f  %.2f m/s" % row)
print("\n  totals:", {k: sum(1 for r in trans_rows if r[1] == k) for k in ('stand->crouch', 'crouch->stand')})
print("\n== jump/flight candidates: clip, takeoff s, landing s, air s, takeoff speed, hip rise m")
by = {}
for row in jump_rows: by.setdefault(row[0], []).append(row)
for clip, rows in by.items():
    print("  %s: %d events" % (clip, len(rows)))
    for row in rows[:40]: print("     %7.2f %7.2f  %.2f s  %.2f m/s  %+.2f m" % row[1:])

# ---- CSV outputs (times in seconds; original LAFAN1 timeline)
import csv
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
with open(os.path.join(OUT, "lafan1_crouch_transitions.csv"), "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["clip", "kind", "start_s", "end_s", "duration_s", "mean_root_speed_mps", "loco_friendly"])
    for c, k, a, b_, d, v in trans_rows:
        lf = c.startswith(("obstacles", "aiming", "multipleActions", "ground")) and 0.2 <= d <= 1.2
        w.writerow([c, k, "%.2f" % a, "%.2f" % b_, "%.2f" % d, "%.2f" % v, int(lf)])
with open(os.path.join(OUT, "lafan1_jump_candidates.csv"), "w", newline="") as fh:
    w = csv.writer(fh, lineterminator="\n")
    w.writerow(["clip", "takeoff_s", "landing_s", "air_s", "takeoff_speed_mps", "hip_rise_m"])
    for c, a, b_, d, v, rise in jump_rows:
        if c.startswith(("jumps", "obstacles", "multipleActions")):
            w.writerow([c, "%.2f" % a, "%.2f" % b_, "%.2f" % d, "%.2f" % v, "%.2f" % rise])
