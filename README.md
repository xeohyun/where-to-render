# Where to Render

**3DGS 지도(.ply) 하나만으로 시각적 위치 추정(ACE)의 학습 시점을 자동으로 고르는 방법.**
실제 사진, 정답 포즈, COLMAP 결과를 쓰지 않고(GT-free) 학습 영상과 포즈를 렌더링해 만들고,
그 영상으로 학습한 ACE가 **실제 사진**의 6-DoF 포즈를 추정한다.

- 웹 데모 : https://xeohyun.github.io/where-to-render/

## 방법

| 단계 | 내용 |
|---|---|
| 1. Map Normalization | 불투명 Gaussian의 PCA + 평평한 Gaussian 법선으로 수직축을 추정하고, 세 가지 기하 단서로 상하 부호를 정한다. 바닥(하위 5%) = 0, 구조물 상단(상위 95%) = 1로 높이를 정규화한다. |
| 2. Candidate Generation | 구조물에 인접한 빈 공간에서 후보 위치를 뽑고, 위치마다 정규화 높이 U[0.1, 1.25]에서 높이 6개를 부여한다. 방향마다 Gaussian 밀도가 높은 쪽을 향하도록 −45°~30° 범위의 피치를 정한다. |
| 3. Render Check & Selection | 색 대신 (1, z, z²)를 넣어 160px로 한 번 렌더링해 픽셀별 불투명도·평균 깊이·깊이 분산을 얻는다. 빈 픽셀 ≤ 5%, 근접 픽셀 ≤ 20%, 상대 깊이 편차 ≤ 0.30을 통과해야 한다. 불합격 방향은 피치를 10°씩 낮춰 다시 검사하고, 연속 4방향 이상 통과한 위치에서 2장씩 렌더링한다. |

구현은 [`sampling/view_sampler.py`](sampling/view_sampler.py)의 `run --preset final`(= 실험 설정 이름 `final3`)이다.

## 결과

위치 오차 중앙값(장면 좌표 단위), 3 seeds 평균. 학습 영상 수·카메라 내부 파라미터·ACE 설정은 모든 방법에서 같다.

| Scene | Ours | Random (same height, level) | Random (full) | Candidate + random dir. | Ours / Random (level) |
|---|---:|---:|---:|---:|---:|
| counter | **0.0531** | 1.8 | 16 | 4.01 | 1/34 |
| bonsai | **0.0223** | 0.879 | 22 | 2.39 | 1/39 |
| kitchen | **0.488** | 4.3 | 13 | 3.48 | 1/9 |
| room | **0.0492** | 0.185 | 23 | 1.33 | 1/4 |
| stump | **5.23** | 43 | 88 | 31 | 1/8 |
| bicycle | **0.205** | 0.89 | 50 | 7.2 | 1/4 |
| garden † | **0.0784** | 0.46 | 10 | 2.93 | 1/6 |
| T&T truck † | **0.0246** | 0.104 | 91 | 10 | 1/4 |
| T&T train † | **7.69** | 14 | 89 | 32 | 1/1.8 |
| LivingRoom | **0.025** | 0.0622 | 0.983 | 1.35 | 1/2 |

† 규칙 설계에 쓰지 않은 held-out 장면(최종 설정을 고를 때는 일부 참고함).
10개 장면 × 3 seeds의 30개 비교 모두에서 세 무작위 기준선보다 오차가 낮았다. 원자료: [`sampling/mcmc/s2_aim_20261009/results.json`](sampling/mcmc/s2_aim_20261009/results.json) (키 `<scene>|<config>|s<seed>|test`).

- 제안 방법은 수직 방향까지 지도에서 추정한다. 무작위 기준선은 COLMAP에서 얻은 수직 방향을 그대로 쓰므로 비교 조건은 제안 방법에 불리하다.
- 실제 test 카메라(정답)는 평가에만 썼다.

## 저장소 구성

이 저장소의 `sampling/`은 [ACE](https://github.com/nianticlabs/ace) 저장소 루트에 두고 실행하도록 만들어져 있다. ACE 코드 자체는 포함하지 않는다.

```
sampling/
  view_sampler.py        시점 선택 알고리즘(run) + 무작위 기준선(random --mode upright|full|pillar)
  common.py              렌더 카메라, ACE 형식 저장, Gaussian 기하 도구
  eval_protocol.py       고정 평가셋 생성·검증(실제 사진 test, LivingRoom 지정 시점 test)
  scenes.json            장면 목록(.ply, COLMAP, 실제 사진 경로; ACE 루트 기준 상대 경로)
  eval_sets/             LivingRoom test 시점(사람이 고른 40개 포즈)
  mcmc/maps/             3DGS-MCMC로 Mip-NeRF 360 지도 학습
  mcmc/s2_aim_20261009/  실험 러너(run.py), 설정(configs_extra.json), 결과(results.json)
  up_estimation/         수직 방향 추정 오차 평가(개발 11개 + held-out 9개 지도)
  figures/               판넬 그림(make_poster_figs.py), 렌더 검사 예시 선정, QR
  demo/                  웹 데모 데이터 내보내기(export_demo.py → docs/)
scripts/run_paper.sh     주요 결과 재현(평가셋 → 시점 선택 → ACE 학습 → 평가)
docs/                    웹 데모(GitHub Pages)
```

## 설치

1. [ACE](https://github.com/nianticlabs/ace)를 받고 README대로 `ace` conda 환경과 DSAC*를 설치한다.
2. [3DGS-MCMC](https://github.com/ubc-vision/3dgs-mcmc)를 받아 별도 conda 환경(예: `3dgs_mcmc`)을 만든다. 렌더러로 쓴다.
3. 이 저장소의 `sampling/`, `scripts/`를 ACE 저장소 루트에 복사한다.
4. 환경 변수를 정한다.

```bash
export GS_REPO=/path/to/3dgs-mcmc
export GS_PYTHON=/path/to/envs/3dgs_mcmc/bin/python
export ACE_PYTHON=/path/to/envs/ace/bin/python
```

## 데이터

ACE 루트 아래에 둔다(`sampling/scenes.json` 경로 기준).

- Mip-NeRF 360 → `datasets/360_v2/<scene>`; 지도는 `sampling/mcmc/maps/run_mcmc_all_scenes.sh`로 학습 → `out/mcmc/mcmc_models/<scene>_mcmc/...`
- Tanks and Temples (3DGS 배포본 `tandt_db`) → `datasets/tandt/{truck,train}`
- SuperSplat LivingRoom → `datasets/LivingRoom/scene.ply`

## 실행

```bash
# 한 장면: 학습 시점 선택만 (3dgs_mcmc env, GPU)
$GS_PYTHON sampling/view_sampler.py run --preset final --scene counter --out_dir out/counter_final --n_train 210 --seed 0 --link_test test

# 논문 주요 결과 전체 (results.json을 새로 쌓으려면 기존 파일을 옮겨 둔다)
bash scripts/run_paper.sh
```

`run.py`는 장면별 카메라(해상도·초점)와 학습 장수를 실제 사진 조건에 맞춰 정한다. 그림은 `sampling/figures/make_poster_figs.py`로 다시 만든다.

## 참고

- E. Brachmann et al., *Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses*, CVPR 2023.
- S. Kheradmand et al., *3D Gaussian Splatting as Markov Chain Monte Carlo*, NeurIPS 2024.
- 데이터: Mip-NeRF 360, Tanks and Temples, SuperSplat LivingRoom. 각 데이터의 라이선스를 따른다.
