"""
eval_protocol.py — 씬별 고정 평가셋(dev/test) 생성·고정·연결. 알고리즘(view_sampler.py)과 독립.
==================================================================================================
원칙 (2026-10-02, 사용자 요청 + 2026-09-29 교차평가에서 확인된 교란 대응)
  * 평가셋은 **한 번 만들고 고정**한다. 이후 알고리즘이 바뀌어도 다시 만들지 않는다(같은 경로에 있으면 freeze가 거부).
    파일마다 sha256을 manifest.json에 남기고 `verify`로 바뀌지 않았는지 확인한다.
  * 평가셋은 **평가 대상 샘플링 정책과 무관한 규칙**으로 만든다 — 2026-09-29에 각 변형이 자기 정책으로 만든 테스트셋으로
    평가하면 "학습셋이 좋아짐"과 "테스트가 쉬워짐"이 섞였다.
  * **dev와 test를 분리**한다: 임계값·하이퍼파라미터 조정은 dev에서만, test는 최종 보고에만 쓴다(과적합 방지).

씬별 구성
  Mip-NeRF 360 (실제 사진 있음): test = 실제 사진 중 COLMAP 포즈 순서 i%8==0 (기존 0e/random/0d 실험과 바이트 동일한
    out/<scene>_0e_mcmc/test를 그대로 고정), dev = i%8==4 실제 사진(test와 겹치지 않음). 실제 사진은 학습에 쓰지 않으므로
    GT-free 원칙과 충돌하지 않는다. 테스트가 실사진이라 "렌더로 학습 → 실사진으로 위치추정"이라는 최종 목표를 직접 잰다.
  SuperSplat (LivingRoom, Street; 실제 사진 없음): 2026-10-02 사용자 결정 — **사용자가 picker로 직접 고른 시점 40장을 test로**
    쓰고 dev는 만들지 않는다(이 두 씬에서는 아무것도 조정하지 않음, 조정은 Mip360 dev에서만). 흐름: `picker`로 HTML 생성 →
    브라우저에서 시점 저장·JSON 내려받기 → `import-poses`가 실제 3DGS로 1600x1064 렌더해 고정. 아래 중립 합성 규칙(freeze_synthetic)은
    만들어 봤으나 Street에서 하늘/허공 쿼리만 나와 폐기했다(datasets/eval_fixed/_unused/, 어떤 평가에도 안 씀).
  (폐기) 중립 합성셋. 위치는 수평 bbox 안 균등 + 높이 s~U[0.15,0.85]
    (바닥 p5 ~ p99 기준), free-space(가우시안 표면거리 > min_offset), yaw 균등, pitch~U[-30,30]도, roll 0, 1600x1064 65도 FOV.
    빈 화면/벽만 보이는 프레임만 제외(void<=0.05, near<=0.5) — 알고리즘의 품질 필터(ray 퍼짐, 선명도 등)는 일부러 쓰지 않는다.
    dev seed 1001, test seed 2002, 각 40장. 한계: "중립"도 하나의 설계 선택이다(높이·pitch 범위).

사용
  python sampling/eval_protocol.py freeze --scene counter        # (3dgs_mcmc env, 합성셋은 GPU 필요)
  python sampling/eval_protocol.py verify --scene counter
  python sampling/eval_protocol.py link --scene counter --split dev --dataset out/<run>   # out/<run>/test -> 고정셋
  python sampling/eval_protocol.py picker --scene livingroom --out <html>                  # 시점 선택 도구(test용)
  python sampling/eval_protocol.py picker --scene livingroom --out <html> --purpose train --target 300
                                                                    # 사람 기준선 학습셋 촬영 → view_sampler.py human
  python sampling/eval_protocol.py import-poses --scene livingroom --poses <json>          # (3dgs_mcmc, GPU)
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
EVAL_ROOT = ROOT / "datasets/eval_fixed"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(split_dir, info):
    files = sorted(p for p in split_dir.rglob("*") if p.is_file() and p.name != "manifest.json")
    info["files"] = {str(p.relative_to(split_dir)): sha256(p) for p in files}
    info["n_frames"] = len(list((split_dir / "rgb").glob("*.png")))
    (split_dir / "manifest.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))


def freeze_mip360(name, cfg):
    from common import load_colmap_poses, write_ace_frame
    from PIL import Image
    base = EVAL_ROOT / name
    poses = load_colmap_poses(str(ROOT / cfg["colmap_sparse"]))
    # test: 기존 실험들이 쓴 실사진 테스트셋을 그대로 고정(새로 만들면 PNG 인코딩 차이가 생길 수 있어 복사)
    src = ROOT / f"out/{name}_0e_mcmc/test"
    n_expect = sum(1 for i in range(len(poses)) if i % 8 == 0)
    if not src.exists():
        # 공개본: 이전 실험 산출물이 없으면 같은 규칙(i%8==0 실제 사진)으로 바로 만든다(PNG 인코딩만 다를 수 있음).
        real = ROOT / cfg["real_images"]
        test = [p for i, p in enumerate(poses) if i % 8 == 0]
        for k, p in enumerate(test):
            img = np.array(Image.open(real / p["name"]).convert("RGB"))
            write_ace_frame(base, "test", k, img, p["R_w2c"], p["t_w2c"], p["focal_px"])
        write_manifest(base / "test", dict(scene=name, split="test", rule="real photos, COLMAP pose order i%8==0",
                                           names=[p["name"] for p in test]))
        src = None
    if src is not None:
        _copy_frozen_test(src, base, poses, n_expect, name)
    real = ROOT / cfg["real_images"]
    dev = [p for i, p in enumerate(poses) if i % 8 == 4]
    for k, p in enumerate(dev):
        img = np.array(Image.open(real / p["name"]).convert("RGB"))
        write_ace_frame(base, "dev", k, img, p["R_w2c"], p["t_w2c"], p["focal_px"])
    write_manifest(base / "dev", dict(scene=name, split="dev", rule="real photos, COLMAP pose order i%8==4",
                                      names=[p["name"] for p in dev]))


def _copy_frozen_test(src, base, poses, n_expect, name):
    """연구 당시 경로: 기존 실험들이 쓴 실사진 테스트셋(out/<scene>_0e_mcmc/test)을 포즈를 확인한 뒤 그대로 복사한다."""
    n_src = len(list((src / "rgb").glob("*.png")))
    assert n_src == n_expect, f"{src}: {n_src}장 != i%8==0 {n_expect}장"
    from common import w2c_to_ace_pose
    for k, p in enumerate(q for i, q in enumerate(poses) if i % 8 == 0):
        got = np.loadtxt(src / "poses" / f"frame_{k:06d}.pose.txt")
        assert np.allclose(got, w2c_to_ace_pose(p["R_w2c"], p["t_w2c"]), atol=1e-5), f"{src} frame {k} 포즈가 {p['name']}와 다름"
    shutil.copytree(src, base / "test", symlinks=False)
    write_manifest(base / "test", dict(scene=name, split="test", rule="real photos, COLMAP pose order i%8==0",
                                       source=str(src.relative_to(ROOT)),
                                       names=[p["name"] for i, p in enumerate(poses) if i % 8 == 0]))


def freeze_synthetic(name, cfg, n=40):
    import view_sampler as vs
    from common import look_at_rotation, write_ace_frame
    R = vs.Renderer(cfg["ply"], cfg["sh_degree"])
    g = R.g
    a = vs.geometry_namespace(seed=0)
    G = vs.s0_scene_geometry(g.get_xyz.detach().cpu().numpy(), g.get_opacity.detach().cpu().numpy().reshape(-1),
                             g.get_scaling.detach().cpu().numpy(), cfg["up"], a)
    up = G["up"]
    h = G["clean"] @ up
    floor, ceil = float(np.percentile(h, 5)), float(np.percentile(h, 99))
    W, H, fov = 1600, 1064, 65.0
    focal = vs.focal_for(W, fov)
    base = EVAL_ROOT / name
    for split, seed in [("dev", 1001), ("test", 2002)]:
        rng = np.random.default_rng(seed)
        k = tries = 0
        log = []
        while k < n:
            tries += 1
            if tries > 200 * n:
                raise SystemExit(f"{name} {split}: {tries}회 시도에도 {n}장을 못 채움")
            p = G["lo"] + rng.random(3) * (G["hi"] - G["lo"])
            p = p - (p @ up) * up + (floor + rng.uniform(0.15, 0.85) * (ceil - floor)) * up
            if not vs.level_position_ok(G, p[None], a)[0]:
                continue
            f = vs.sweep_forwards(up, 1, rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(-30, 30)))[0]
            Rw = look_at_rotation(f, up); tw = -Rw @ p
            cam = R.camera(Rw, tw, focal, W, H)
            m = vs.frame_metrics(*R.stats(cam, Rw, tw), G["near_dist"])
            if m["void_frac"] > 0.05 or m["near_frac"] > 0.5:
                continue
            write_ace_frame(base, split, k, R.color(cam, 1.0), Rw, tw, focal)
            log.append(dict(frame=k, pos=p.tolist(), void=m["void_frac"], near=m["near_frac"], relstd=m["relstd"]))
            k += 1
        write_manifest(base / split, dict(scene=name, split=split, seed=seed, tries=tries,
                                          rule="neutral synthetic: uniform horiz pos, s~U[0.15,0.85] of [p5,p99] height, free space, "
                                               "yaw~U, pitch~U[-30,30], roll 0, 1600x1064 65deg; keep void<=0.05 & near<=0.5",
                                          floor_p5=floor, ceil_p99=ceil, frames=log))
        print(f"{name} {split}: {k}장 ({tries}회 시도)")


def read_ply_fields(path):
    import re
    with open(path, "rb") as f:
        head = b""
        while True:
            line = f.readline()
            head += line
            if line.strip() == b"end_header":
                break
        off = f.tell()
    t = head.decode("ascii", "ignore")
    n = int(re.search(r"element vertex (\d+)", t).group(1))
    props = re.findall(r"property float (\S+)", t)
    return np.fromfile(path, dtype=np.dtype([(p, "<f4") for p in props]), count=n, offset=off)


PICKER_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
html,body{margin:0;height:100%;background:#111;color:#eee;font:13px/1.4 system-ui,sans-serif;overflow:hidden}
#wrap{position:absolute;inset:0 0 120px 0;display:flex;align-items:center;justify-content:center}
canvas{background:#000;outline:1px solid #555}
#hud{position:absolute;left:10px;top:8px;background:rgba(0,0,0,.65);padding:8px 10px;border-radius:6px;white-space:pre}
#help{position:absolute;right:10px;top:8px;background:rgba(0,0,0,.65);padding:8px 10px;border-radius:6px;white-space:pre}
#strip{position:absolute;left:0;right:0;bottom:0;height:116px;background:#1b1b1b;overflow-x:auto;white-space:nowrap;padding:4px}
#strip div{display:inline-block;margin-right:4px;text-align:center;font-size:11px;cursor:pointer;border:2px solid transparent}
#strip div.sel{border-color:#4cf}
#strip img{display:block;height:86px}
#flash{position:absolute;inset:0;background:#fff;opacity:0;pointer-events:none;transition:opacity .25s}
</style></head><body>
<div id="wrap"><canvas id="c"></canvas></div><div id="hud"></div>
<div id="help">클릭: 마우스 시점 잠금(ESC 해제)
마우스: 좌우/상하 보기 (roll은 항상 0)
W/A/S/D: 수평 이동   Q/E: 아래/위
Shift: 4배 빠르게   휠: 기본 속도
Space: 현재 시점 저장__HELP_EXTRA__
Z: 마지막 저장 취소   Backspace: 선택한 저장 삭제
썸네일 클릭: 그 시점으로 이동
X: 포즈 JSON 내려받기 (__POSEFILE__)
R: 시작 위치로
화면 테두리 = 최종 렌더 구도(1600x1064, 수평 FOV 65도)
점 스플랫 근사라 반투명/얇은 부분은 실제 렌더와 다를 수 있음</div>
<div id="strip"></div><div id="flash"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
const META=__META__;
function b64(s,T){const bin=atob(s);const u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return new T(u.buffer);}
const P=b64(META.pos,Uint16Array), C=b64(META.col,Uint8Array), S=b64(META.size,Uint8Array);
const N=S.length, lo=META.lo, hi=META.hi;
const pos=new Float32Array(N*3), rad=new Float32Array(N);
for(let i=0;i<N;i++){for(let k=0;k<3;k++)pos[3*i+k]=lo[k]+P[3*i+k]/65535*(hi[k]-lo[k]);
  rad[i]=Math.exp(META.lr0+S[i]/255*(META.lr1-META.lr0));}
const up=new THREE.Vector3(...META.up).normalize();
let a=new THREE.Vector3(1,0,0); if(Math.abs(up.x)>=0.9)a=new THREE.Vector3(0,1,0);
const e1=a.clone().sub(up.clone().multiplyScalar(a.dot(up))).normalize(), e2=new THREE.Vector3().crossVectors(up,e1);
const AR=META.width/META.height, VFOV=2*Math.atan(Math.tan(META.hfov*Math.PI/360)/AR)*180/Math.PI;
const canvas=document.getElementById('c');
const renderer=new THREE.WebGLRenderer({canvas,antialias:false,preserveDrawingBuffer:true});
renderer.setPixelRatio(1);
const scene=new THREE.Scene(); scene.background=new THREE.Color(1,1,1);
const cam=new THREE.PerspectiveCamera(VFOV,AR,0.01,1e5); cam.up.copy(up);
const geo=new THREE.BufferGeometry();
geo.setAttribute('position',new THREE.BufferAttribute(pos,3));
geo.setAttribute('rgba',new THREE.BufferAttribute(C,4,true));
geo.setAttribute('rad',new THREE.BufferAttribute(rad,1));
const mat=new THREE.ShaderMaterial({uniforms:{proj:{value:1}},
 vertexShader:`attribute float rad; attribute vec4 rgba; uniform float proj; varying vec4 vC;
  void main(){vec4 mv=modelViewMatrix*vec4(position,1.0); gl_Position=projectionMatrix*mv; vC=rgba;
   gl_PointSize=clamp(2.0*rad*proj/max(-mv.z,1e-4),1.5,96.0);}`,
 fragmentShader:`varying vec4 vC; void main(){vec2 d=gl_PointCoord-0.5; float r2=dot(d,d)*4.0;
   float w=vC.a*exp(-2.0*r2); if(w<0.25) discard; gl_FragColor=vec4(vC.rgb,1.0);}`});
scene.add(new THREE.Points(geo,mat));
let P0=new THREE.Vector3(...META.start), cur=P0.clone(), yaw=0, pitch=0, speed=META.speed, caps=[], sel=-1;
const KEY='picker_'+META.scene+(META.purpose==='train'?'_train':'');
let rec=false, lastCap=null, lastT=0, storageOK=true;
try{const s=JSON.parse(localStorage.getItem(KEY)||'null'); if(s&&s.caps)caps=s.caps;}catch(e){}
function fwd(y,p){const h=e1.clone().multiplyScalar(Math.cos(y)).add(e2.clone().multiplyScalar(Math.sin(y)));
  return h.multiplyScalar(Math.cos(p)).add(up.clone().multiplyScalar(Math.sin(p))).normalize();}
function resize(){const W=window.innerWidth,H=window.innerHeight-120;let w=W,h=W/AR;if(h>H){h=H;w=H*AR;}
  renderer.setSize(Math.floor(w),Math.floor(h),true); mat.uniforms.proj.value=(h/2)/Math.tan(VFOV*Math.PI/360);}
window.addEventListener('resize',resize); resize();
const keys={};
window.addEventListener('keydown',e=>{keys[e.code]=true; if(e.code==='Space'){e.preventDefault();capture(false);}
  if(e.code==='KeyT'&&META.purpose==='train'){rec=!rec;lastCap=null;}
  if(e.code==='KeyZ'&&caps.length){caps.pop();sel=-1;save();strip();}
  if(e.code==='Backspace'&&sel>=0){e.preventDefault();caps.splice(sel,1);sel=-1;save();strip();}
  if(e.code==='KeyX')exportJSON(); if(e.code==='KeyR'){cur.copy(P0);yaw=0;pitch=0;}});
window.addEventListener('keyup',e=>{keys[e.code]=false;});
canvas.addEventListener('click',()=>canvas.requestPointerLock());
document.addEventListener('mousemove',e=>{if(document.pointerLockElement!==canvas)return;
  yaw-=e.movementX*0.0025; pitch=Math.max(-1.48,Math.min(1.48,pitch-e.movementY*0.0025));});
window.addEventListener('wheel',e=>{speed*=e.deltaY<0?1.15:1/1.15;});
function save(){try{localStorage.setItem(KEY,JSON.stringify({caps}));storageOK=true;}catch(e){storageOK=false;}}
function thumb(){const t=document.createElement('canvas');t.height=META.thumb_h;t.width=Math.round(META.thumb_h*AR);
  t.getContext('2d').drawImage(canvas,0,0,t.width,t.height);return t.toDataURL('image/jpeg',0.7);}
function capture(auto){const f=fwd(yaw,pitch);
  caps.push({position:cur.toArray(),forward:f.toArray(),yaw_deg:yaw*180/Math.PI,pitch_deg:pitch*180/Math.PI,
    s:(cur.dot(up)-META.floor)/(META.top-META.floor),t:new Date().toISOString(),auto:auto,thumb:thumb()});
  lastCap={p:cur.clone(),f:f.clone()};lastT=performance.now();
  save();strip();const fl=document.getElementById('flash');fl.style.opacity=.6;setTimeout(()=>fl.style.opacity=0,80);}
function strip(){const el=document.getElementById('strip');el.innerHTML='';
  caps.forEach((c,i)=>{const d=document.createElement('div');if(i===sel)d.className='sel';
   d.innerHTML=`<img src="${c.thumb}"><span>#${i+1} s=${c.s.toFixed(2)} p=${c.pitch_deg.toFixed(0)}</span>`;
   d.onclick=()=>{sel=i;cur.fromArray(c.position);yaw=c.yaw_deg*Math.PI/180;pitch=c.pitch_deg*Math.PI/180;strip();};el.appendChild(d);});
  el.scrollLeft=el.scrollWidth;}
function exportJSON(){const out={scene:META.scene,purpose:META.purpose,up:META.up,hfov_deg:META.hfov,width:META.width,height:META.height,
   convention:'.ply world coords; camera looks along forward, roll 0 w.r.t. up (view_sampler look_at_rotation)',
   captures:caps.map(({thumb,...r})=>r)};
  const blob=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});const u=URL.createObjectURL(blob);
  const l=document.createElement('a');l.href=u;l.download='__POSEFILE__';document.body.appendChild(l);l.click();l.remove();}
let last=performance.now();
function loop(now){const dt=Math.min(0.1,(now-last)/1000);last=now;
  const h=fwd(yaw,0), r=new THREE.Vector3().crossVectors(h,up), v=speed*dt*(keys.ShiftLeft||keys.ShiftRight?4:1);
  if(keys.KeyW)cur.addScaledVector(h,v); if(keys.KeyS)cur.addScaledVector(h,-v);
  if(keys.KeyD)cur.addScaledVector(r,v); if(keys.KeyA)cur.addScaledVector(r,-v);
  if(keys.KeyE)cur.addScaledVector(up,v); if(keys.KeyQ)cur.addScaledVector(up,-v);
  cam.position.copy(cur); cam.lookAt(cur.clone().add(fwd(yaw,pitch))); renderer.render(scene,cam);
  if(rec&&now-lastT>200){const f=fwd(yaw,pitch);
    if(!lastCap||cur.distanceTo(lastCap.p)>=META.rec_step||f.angleTo(lastCap.f)>=META.rec_deg*Math.PI/180)capture(true);}
  const s=(cur.dot(up)-META.floor)/(META.top-META.floor);
  document.getElementById('hud').textContent=`${META.scene} [${META.purpose}]  저장 ${caps.length}/${META.target}${rec?'  ● 자동 촬영 중':''}${storageOK?'':'\n⚠ 브라우저 자동 저장 실패 — X로 내려받으세요'}\n높이 s=${s.toFixed(2)} (바닥 p5=0, 상위5%=1)\npitch ${(pitch*180/Math.PI).toFixed(0)}°  yaw ${((yaw*180/Math.PI)%360).toFixed(0)}°\n속도 ${speed.toPrecision(2)}/s  점 ${N.toLocaleString()}개`;
  requestAnimationFrame(loop);}
strip(); requestAnimationFrame(loop);
</script></body></html>"""


def cmd_picker(a):
    """브라우저 시점 선택 도구(HTML 한 파일) 생성. 실제 Gaussian을 색(SH DC)·크기·opacity 반영한 원형 스플랫으로 그린다."""
    import base64
    import view_sampler as vs
    from common import remove_floater_points
    cfg = vs.load_scene_cfg(a.scene)
    rng = np.random.default_rng(0)
    v = read_ply_fields(cfg["ply"])
    op = 1 / (1 + np.exp(-v["opacity"]))
    keep = np.where(op > 0.2)[0]
    if len(keep) > a.pre_points:
        keep = rng.choice(keep, a.pre_points, replace=False)
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)[keep].astype(np.float64)
    xyz, m = remove_floater_points(xyz, k=20, std_ratio=2.0)
    keep = keep[m]
    if len(keep) > a.max_points:
        sub = rng.choice(len(keep), a.max_points, replace=False)
        keep, xyz = keep[sub], xyz[sub]
    sc = np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1)[keep])
    top2 = np.sort(sc, 1)[:, 1:]
    rad = np.sqrt(top2[:, 0] * top2[:, 1])
    rgb = np.clip(0.5 + 0.28209479177387814 * np.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], 1)[keep], 0, 1)
    col = np.concatenate([rgb, op[keep][:, None]], 1)
    lo, hi = xyz.min(0), xyz.max(0)
    q = np.round((xyz - lo) / np.maximum(hi - lo, 1e-9) * 65535).astype("<u2")
    lr = np.log(np.maximum(rad, 1e-6))
    lr0, lr1 = np.percentile(lr, [0.5, 99.5])
    qs = np.round(np.clip((lr - lr0) / (lr1 - lr0), 0, 1) * 255).astype(np.uint8)
    up = cfg["up"]
    h = xyz @ up
    floor, top = float(np.percentile(h, 5)), float(np.percentile(h, 95))
    horiz = np.median(xyz - np.outer(h, up), 0)
    start = horiz + (floor + a.start_s * (top - floor)) * up
    speed = float(np.linalg.norm(hi - lo) / 40)
    meta = dict(scene=a.scene, purpose=a.purpose, up=up.tolist(), hfov=65.0, width=1600, height=1064, floor=floor, top=top,
                start=start.tolist(), speed=speed, target=a.target, thumb_h=86 if a.purpose == "test" else 40,
                rec_step=a.rec_step if a.rec_step else speed / 2, rec_deg=a.rec_deg,
                lo=lo.tolist(), hi=hi.tolist(), lr0=float(lr0), lr1=float(lr1),
                pos=base64.b64encode(q.tobytes()).decode(), col=base64.b64encode(np.round(col * 255).astype(np.uint8).tobytes()).decode(),
                size=base64.b64encode(qs.tobytes()).decode())
    posefile = f"{a.scene}_{a.purpose}_poses.json"
    extra = ("" if a.purpose == "test" else
             "\nT: 자동 촬영 켜기/끄기(이동 %.3g 또는 회전 %g도마다 1장)\n학습셋 = 사람이 직접 촬영하듯 자연스럽게 돌아다니며"
             % (meta["rec_step"], a.rec_deg))
    html = (PICKER_HTML.replace("__META__", json.dumps(meta)).replace("__TITLE__", f"{a.scene} 시점 선택 ({a.purpose})")
            .replace("__POSEFILE__", posefile).replace("__HELP_EXTRA__", extra))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(html)
    print(f"{a.out}: 점 {len(keep):,}개, {Path(a.out).stat().st_size / 1e6:.1f}MB, 바닥 p5={floor:.3f}, 상위5%={top:.3f}")


def cmd_import_poses(a):
    """picker에서 내려받은 포즈 JSON을 실제 3DGS 렌더러로 1600x1064 렌더해 고정 test로 만든다(사람이 고른 시점 = 알고리즘과 독립)."""
    import view_sampler as vs
    from common import look_at_rotation, write_ace_frame
    cfg = vs.load_scene_cfg(a.scene)
    J = json.loads(Path(a.poses).read_text())
    assert J["scene"] == a.scene, f"포즈 파일 씬 {J['scene']} != {a.scene}"
    up = np.array(J["up"], dtype=np.float64)
    assert np.allclose(up / np.linalg.norm(up), cfg["up"], atol=1e-6), "포즈 파일 up이 scenes.json과 다름"
    base = EVAL_ROOT / a.scene
    if (base / "test").exists():
        raise SystemExit(f"{base / 'test'} 가 이미 있습니다 — 고정 평가셋은 다시 만들지 않습니다.")
    R = vs.Renderer(cfg["ply"], cfg["sh_degree"])
    W, H = int(J["width"]), int(J["height"])
    focal = vs.focal_for(W, float(J["hfov_deg"]))
    g = R.g
    ga = vs.geometry_namespace(seed=0)
    G = vs.s0_scene_geometry(g.get_xyz.detach().cpu().numpy(), g.get_opacity.detach().cpu().numpy().reshape(-1),
                             g.get_scaling.detach().cpu().numpy(), cfg["up"], ga)
    frames = []
    for k, c in enumerate(J["captures"]):
        p = np.array(c["position"], dtype=np.float64)
        Rw = look_at_rotation(np.array(c["forward"], dtype=np.float64), cfg["up"]); tw = -Rw @ p
        cam = R.camera(Rw, tw, focal, W, H)
        write_ace_frame(base, "test", k, R.color(cam, 1.0), Rw, tw, focal)
        m = vs.frame_metrics(*R.stats(cam, Rw, tw), G["near_dist"])
        frames.append(dict(frame=k, **{kk: c[kk] for kk in ("position", "forward", "yaw_deg", "pitch_deg", "s")}, **m))
    write_manifest(base / "test", dict(scene=a.scene, split="test", rule="user-picked viewpoints (eval_protocol picker), rendered by 3DGS",
                                       poses_file=str(Path(a.poses).resolve()), poses_sha256=sha256(a.poses), hfov_deg=J["hfov_deg"],
                                       width=W, height=H, frames=frames))
    from PIL import Image
    ims = sorted((base / "test" / "rgb").glob("*.png"))
    cols = 8; tw_, th_ = 200, int(200 * H / W)
    sheet = Image.new("RGB", (cols * tw_, ((len(ims) + cols - 1) // cols) * th_))
    for i, p in enumerate(ims):
        sheet.paste(Image.open(p).resize((tw_, th_)), ((i % cols) * tw_, (i // cols) * th_))
    sheet.save(base / "test_contact_sheet.jpg", quality=85)
    warn = [f["frame"] for f in frames if f["void_frac"] > 0.05]
    print(f"{a.scene}: test {len(frames)}장 고정, 미리보기 {base / 'test_contact_sheet.jpg'}"
          + (f", 미재구성 5% 초과 프레임 {warn}(확인 권장)" if warn else ""))


def cmd_freeze(a):
    import view_sampler as vs
    cfg = vs.load_scene_cfg(a.scene)
    base = EVAL_ROOT / a.scene
    if base.exists():
        raise SystemExit(f"{base} 가 이미 있습니다 — 고정 평가셋은 다시 만들지 않습니다(verify로 확인).")
    base.mkdir(parents=True)
    (freeze_mip360 if cfg.get("colmap_sparse") else freeze_synthetic)(a.scene, cfg)
    cmd_verify(a)


def cmd_verify(a):
    base = EVAL_ROOT / a.scene
    for split in ["dev", "test"]:
        if not (base / split).exists():
            print(f"{a.scene}/{split}: 없음(이 씬은 {split} 없이 운영)"); continue
        m = json.loads((base / split / "manifest.json").read_text())
        bad = [f for f, h in m["files"].items() if sha256(base / split / f) != h]
        print(f"{a.scene}/{split}: {m['n_frames']}장, 해시 불일치 {len(bad)}개" + (f" {bad[:3]}" if bad else ""))
        if bad:
            raise SystemExit(1)


def cmd_link(a):
    """학습 데이터셋 폴더의 test를 고정 평가셋(dev 또는 test) 심볼릭 링크로 만든다. ACE는 <dataset>/test를 읽는다."""
    ds = Path(a.dataset)
    dst = ds / "test"
    if dst.exists() or dst.is_symlink():
        if not (dst.is_symlink() and a.force):
            raise SystemExit(f"{dst} 가 이미 있습니다(심볼릭 링크면 --force로 교체)")
        dst.unlink()
    dst.symlink_to((EVAL_ROOT / a.scene / a.split).resolve())
    print(f"{dst} -> {dst.resolve()}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ["freeze", "verify"]:
        p = sub.add_parser(c); p.add_argument("--scene", required=True)
    p = sub.add_parser("link"); p.add_argument("--scene", required=True); p.add_argument("--dataset", required=True)
    p.add_argument("--split", choices=["dev", "test"], required=True); p.add_argument("--force", action="store_true")
    p = sub.add_parser("picker", help="브라우저 시점 선택 HTML 생성")
    p.add_argument("--scene", required=True); p.add_argument("--out", required=True)
    p.add_argument("--max_points", type=int, default=1_000_000); p.add_argument("--pre_points", type=int, default=3_000_000)
    p.add_argument("--start_s", type=float, default=0.5); p.add_argument("--target", type=int, default=40)
    p.add_argument("--purpose", choices=["test", "train"], default="test",
                   help="test: 고정 평가셋용(import-poses). train: 사람 기준선 학습셋용(view_sampler.py human), 자동 촬영(T) 지원")
    p.add_argument("--rec_step", type=float, default=None, help="train 자동 촬영 이동 간격(기본: 기본 이동속도/2)")
    p.add_argument("--rec_deg", type=float, default=15.0, help="train 자동 촬영 회전 간격(도)")
    p = sub.add_parser("import-poses", help="picker 포즈 JSON -> 3DGS 렌더 -> 고정 test")
    p.add_argument("--scene", required=True); p.add_argument("--poses", required=True)
    a = ap.parse_args()
    dict(freeze=cmd_freeze, verify=cmd_verify, link=cmd_link, picker=cmd_picker, **{"import-poses": cmd_import_poses})[a.cmd](a)


if __name__ == "__main__":
    main()
