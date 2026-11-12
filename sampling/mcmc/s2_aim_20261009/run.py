"""S2(기둥 높이 층 + 자동 pitch) 계열 설정을 학습·평가하는 범용 러너. manifest.json 참고. GPU 1 순차.
사용: python run.py --tag dev1 --configs aim_rs,aim --scenes counter,... --seeds 0 --evals dev
      evals: dev(실사진 i%8==4) / test(실사진 i%8==0 또는 SuperSplat 고정 test) / rtest(Mip360 같은 포즈 렌더 test)
      policy 이름이 random_upright / random_full이면 view_sampler random으로 만든다."""
import argparse, json, os, statistics as S, subprocess, time, traceback, fcntl
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; RUN = Path(__file__).resolve().parent; OUT = ROOT / 'output/s2_aim_20261009'
# 두 conda 환경의 python 경로(README 참고): 3DGS 렌더용(3dgs_mcmc), ACE 학습·평가용(ace)
GS = os.environ.get('GS_PYTHON', 'python'); ACE = os.environ.get('ACE_PYTHON', 'python')
CONFIGS = {
    'aim_rs': ['--s1', 'pillar', '--s2', 'aim', '--views_per_point', '2', '--n_candidates', '2000', '--max_relstd', '0.30'],
    'aim': ['--s1', 'pillar', '--s2', 'aim', '--views_per_point', '2', '--n_candidates', '2000'],
}
# random 계열 추가 인자(E4b: 완전도 필터 + 지도에서 추정한 바닥 기준 높이 제한 s in [0, 1.25] — 알고리즘의 높이 층 범위와 같음)
RANDOM_EXTRA = {'random_filteredH': ('filtered', ['--s_min', '0', '--s_max', '1.25']),
                'random_filteredP': ('filtered', ['--pos_source', 'pillar']),
                'random_filteredPA': ('filtered', ['--pos_source', 'pillar', '--pitch_source', 'aim']),
                'random_filteredPV': ('filtered', ['--pos_source', 'pillar', '--views_per_pos', '4']),
                'random_pano': ('filtered', ['--pos_source', 'pillar', '--pano_min', '6']),
                'random_filteredPF': ('filtered', ['--pos_source', 'pillar', '--far_check']),
                'random_pillarUp': ('pillar', ['--pillar_dir', 'upright'])}  # 2026-10-29: 알고리즘 위치 후보 + 구면 균등 시선, roll 0  # E4h: E4c + 근접샷(p90 깊이) 기준  # E4g: 둘레 완전도(12방향 중 6 이상) 위치 + 깨끗한 방향 하나 무작위  # E4e: + 같은 위치 겹치는 다중 시점  # E4d: + aim pitch  # E4c: 구조물 근처 위치 + 무작위 방향 + 완전도 필터
EXTRA = RUN / 'configs_extra.json'  # 이후 반복에서 추가하는 설정(파일로 기록)
if EXTRA.exists():
    CONFIGS.update(json.loads(EXTRA.read_text()))
OUT.mkdir(parents=True, exist_ok=True); (RUN / 'logs').mkdir(exist_ok=True)
RESF = RUN / 'results.json'


def load_res():
    return json.loads(RESF.read_text()) if RESF.exists() else {}


def save_res(k, v):
    with (RUN / 'results.lock').open('w') as lk:  # GPU 두 개 워커가 동시에 쓰므로 잠금
        fcntl.flock(lk, fcntl.LOCK_EX)
        R = load_res(); R[k] = v; (RUN / 'results.tmp').write_text(json.dumps(R, indent=1)); (RUN / 'results.tmp').replace(RESF)


def status(k, state, **kw):
    with (RUN / 'status.lock').open('w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        f = RUN / 'status.json'; st = json.loads(f.read_text()) if f.exists() else {}
        st[k] = dict(state=state, t=time.strftime('%m-%d %H:%M:%S'), **kw)
        (RUN / 'status.tmp').write_text(json.dumps(st, indent=1)); (RUN / 'status.tmp').replace(f)


def free_mb():
    out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    return {int(l.split(',')[0]): int(l.split(',')[1]) for l in out.strip().splitlines()}


FIXED_GPU = False


def pick_gpu(pref, need_mb=14000):
    """다른 사용자 작업이 GPU 메모리를 쓰고 있으므로, 여유 메모리가 need_mb 이상인 GPU가 생길 때까지 기다린다(선호 GPU 우선)."""
    while True:
        f = free_mb()
        for g in [pref] + ([] if FIXED_GPU else [g for g in f if g != pref]):
            if f.get(g, 0) >= need_mb:
                return g
        time.sleep(60)


def sh(key, cmd, gpu):
    """OOM(다른 사용자 메모리 증가)이면 실패 로그를 남기고 기다렸다 최대 3회 재시도."""
    for attempt in range(4):
        g = pick_gpu(gpu)
        log = RUN / f'logs/{key}.log' if attempt == 0 else RUN / f'logs/{key}.retry{attempt}.log'
        with log.open('w') as f:
            p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONNOUSERSITE='1'))
        if not p.returncode:
            return
        if 'OutOfMemoryError' not in log.read_text():
            break
        time.sleep(120)
    raise RuntimeError(f'{key} exit {p.returncode}')


def cam(s):
    if s in ('livingroom', 'street'):
        return ['--width', '1600', '--height', '1064', '--fov_deg', '65'], 300
    from PIL import Image
    d = ROOT / f'out/{s}_real/train'
    W, H = Image.open(sorted((d / 'rgb').glob('*.png'))[0]).size
    f = float(sorted((d / 'calibration').glob('*.txt'))[0].read_text())
    return ['--width', str(W), '--height', str(H), '--focal_px', str(f)], len(list((d / 'rgb').glob('*.png')))


def eval_root(s, split):
    if split == 'rtest':
        return ROOT / f'out/{s}_dg_rtest'
    d = ROOT / f'out/{s}_pd_eval_{split}'
    if not d.exists():
        d.mkdir(); (d / 'test').symlink_to((ROOT / f'datasets/eval_fixed/{s}/{split}').resolve())
    return d


def evaluate(key, root, head, session, gpu):
    if key in load_res():
        return
    sh(f'test_{session}', [ACE, '-u', 'test_ace.py', str(root), str(head), '--session', session], gpu)
    rows = [l.split() for l in (Path(head).parent / f'poses_{Path(root).name}_{session}.txt').read_text().splitlines() if l.strip()]
    r = [float(x[8]) for x in rows]; t = [float(x[9]) for x in rows]
    save_res(key, dict(n=len(r), med_rot=S.median(r), med_trans=S.median(t),
                       acc_5cm5deg=100 * sum(a < 5 and b < .05 for a, b in zip(r, t)) / len(r),
                       acc_10cm10deg=100 * sum(a < 10 and b < .1 for a, b in zip(r, t)) / len(r),
                       acc_25cm5deg=100 * sum(a < 5 and b < .25 for a, b in zip(r, t)) / len(r)))


# 이전 실험에서 같은 코드·같은 조건으로 이미 학습한 기준선 head(재학습 없이 같은 평가 파이프라인으로 다시 평가)
PRIOR = {('random_upright', 0, 'mip'): 'output/algo_vs_random_20261008/{s}_avr_random_upright_s0.pt',
         ('random_full', 0, 'mip'): 'output/rank_check_v2_20261002/{s}__random_full.pt'}
for k in range(3):
    for pol in ('random_upright', 'random_full'):
        PRIOR[(pol, k, 'ss')] = 'output/algo_vs_random_20261008/{s}_avr_' + pol + f'_s{k}.pt'


N_OVERRIDE = None  # --n_train: 학습 장수를 바꾼 실험(이름·결과 키에 @N<n> 접미사)
N_SCALE = None  # --n_scale: 씬별 기본 장수의 배수(2026-10-24 발표 피드백 4, 이름·결과 키에 @x<배수> 접미사)


def tag_of(cfg):
    if N_SCALE is not None:
        return f'{cfg}@x{N_SCALE:g}'
    return cfg if N_OVERRIDE is None else f'{cfg}@N{N_OVERRIDE}'


def build(s, cfg, seed, gpu):
    ca, n = cam(s)
    if N_OVERRIDE is not None:
        n = N_OVERRIDE
    if N_SCALE is not None:
        n = int(round(n * N_SCALE))
    name = f'{s}_s2_{tag_of(cfg).replace("@", "_")}_s{seed}'; d = ROOT / f'out/{name}'; head = OUT / f'{name}.pt'
    if head.exists():
        return head
    pr = None if (N_OVERRIDE is not None or N_SCALE is not None) else PRIOR.get((cfg, seed, 'ss' if s in ('livingroom', 'street') else 'mip'))
    if pr and (ROOT / pr.format(s=s)).exists():
        head.symlink_to((ROOT / pr.format(s=s)).resolve())
        return head
    got = [len(list((d / 'train' / k).glob('*'))) for k in ('rgb', 'poses', 'calibration')] if (d / 'run_args.json').exists() else None
    if got == [n] * 3:  # 렌더는 끝났고 학습만 실패(OOM 등)한 경우 같은 데이터셋으로 학습만 다시
        status(name, 'ace_train'); sh(f'{name}_train', [ACE, '-u', 'train_ace.py', str(d), str(head)], gpu); status(name, 'built')
        return head
    status(name, 'render')
    base = ['--scene', s, '--out_dir', str(d), '--n_train', str(n), '--seed', str(seed), '--clean'] + ca
    if cfg.startswith('random_'):
        mode, extra = RANDOM_EXTRA.get(cfg, (cfg.split('_')[1], []))
        sh(f'{name}_render', [GS, '-u', 'sampling/view_sampler.py', 'random', '--mode', mode] + base + extra, gpu)
    else:
        sh(f'{name}_render', [GS, '-u', 'sampling/view_sampler.py', 'run'] + base + CONFIGS[cfg], gpu)
    got = [len(list((d / 'train' / k).glob('*'))) for k in ('rgb', 'poses', 'calibration')]
    assert got == [n] * 3, f'{name}: {got} != {n}'
    status(name, 'ace_train'); sh(f'{name}_train', [ACE, '-u', 'train_ace.py', str(d), str(head)], gpu)
    status(name, 'built')
    return head


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True); ap.add_argument('--configs', required=True); ap.add_argument('--scenes', required=True)
    ap.add_argument('--seeds', default='0'); ap.add_argument('--evals', default='dev'); ap.add_argument('--gpu', type=int, default=1)
    ap.add_argument('--fixed_gpu', action='store_true', help='여유 메모리를 기다리되 다른 GPU로 옮기지 않음(워커 2개 병렬용)')
    ap.add_argument('--n_train', type=int, default=None, help='학습 장수 덮어쓰기(장수 스케일링 실험)')
    ap.add_argument('--n_scale', type=float, default=None, help='씬별 기본 장수(실사진 학습 장수/300)의 배수')
    a = ap.parse_args()
    FIXED_GPU = a.fixed_gpu
    N_OVERRIDE = a.n_train
    N_SCALE = a.n_scale
    lk = (RUN / f'run_{a.tag}.lock').open('w'); fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for seed in [int(x) for x in a.seeds.split(',')]:
        for s in a.scenes.split(','):
            for cfg in a.configs.split(','):
                try:
                    head = build(s, cfg, seed, a.gpu)
                    for ev in a.evals.split(','):
                        if s in ('livingroom', 'street') and ev != 'test':
                            continue
                        t = tag_of(cfg)
                        evaluate(f'{s}|{t}|s{seed}|{ev}', eval_root(s, ev), head, f'{s}_s2_{t.replace("@", "_")}_s{seed}_{ev}', a.gpu)
                    status(f'{s}_s2_{tag_of(cfg)}_s{seed}', 'done')
                except Exception as e:
                    status(f'{s}_s2_{tag_of(cfg)}_s{seed}', 'failed', error=str(e)); traceback.print_exc()
    print(f'done_{a.tag}', flush=True)
