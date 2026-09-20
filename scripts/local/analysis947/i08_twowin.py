import sys, os, json, math, time
from pathlib import Path
import numpy as np, torch, zarr
REPO=Path('/Volumes/extSSD/kaggle/biohub/replay_assets/support_pack/repo')
sys.path.insert(0,str(REPO/'scripts')); sys.path.insert(0,str(REPO/'src'))
import torch.nn.functional as F
from predict_unet_transformer import load_model, _load_frame, _detect_cells_pooled, pool_kernel_from_um
from biohub_tracking.io import open_dataset

dev=torch.device('mps')
W=2
model,ws_,dsx = load_model(Path('/Volumes/extSSD/kaggle/biohub/replay_assets/support_pack/weights/unet_transformer/split_0/edge_predictor_best.pth'), dev)
print('window',ws_,'ds',dsx)
downsample=(1,4,4)
vol=sys.argv[1] if len(sys.argv)>1 else '6bba_7af54fde'
T0=int(sys.argv[2]) if len(sys.argv)>2 else 20
T1=int(sys.argv[3]) if len(sys.argv)>3 else 40
p=Path(f'/Volumes/extSSD/kaggle/biohub/raw/train/{vol}.zarr')
ds=open_dataset(p, normalize=False, load_image=False, downsample=downsample)
zarr_arr=zarr.open_group(str(ds.zarr_path),mode='r')['0']
q_low=float(ds.quantiles['0.001']); q_high=float(ds.quantiles['0.999'])
target_shape=list(ds.image_shape[1:])
voxel=tuple(s*d for s,d in zip(ds.scale,downsample))
pool_k=pool_kernel_from_um(5.0,voxel)
print('voxel',voxel,'pool_k',pool_k,'shape',ds.image_shape)
DET=0.965

@torch.no_grad()
def window_det(ws):
    imgs=torch.stack([_load_frame(zarr_arr,t,target_shape,downsample) for t in range(ws,ws+W)])
    imgs=((imgs-q_low)/(q_high-q_low+1e-6)).clamp(0.0)
    imgs=imgs.unsqueeze(0).to(dev)
    unet_out,det=model.encode(imgs)
    nv=1
    for dims in [(-1,),(-2,),(-2,-1)]:
        _,d2=model.encode(imgs.flip(dims))
        for f in range(W): det[f]=det[f]+d2[f].flip(dims)
        nv+=1
    for k in (1,3):
        _,d2=model.encode(torch.rot90(imgs,k,dims=(-2,-1)))
        for f in range(W): det[f]=det[f]+torch.rot90(d2[f],-k,dims=(-2,-1))
        nv+=1
    _,d2=model.encode(imgs.transpose(-1,-2))
    for f in range(W): det[f]=det[f]+d2[f].transpose(-1,-2)
    nv+=1
    _,d2=model.encode(torch.rot90(imgs,1,dims=(-2,-1)).transpose(-1,-2))
    for f in range(W): det[f]=det[f]+torch.rot90(d2[f].transpose(-1,-2),-1,dims=(-2,-1))
    nv+=1
    return [ (det[f]/nv).cpu() for f in range(W) ]

def peaks(logit,t):
    return _detect_cells_pooled(logit.to(dev)[0], t, DET, pool_k)

def match(a,b,tol=1e-9):
    # a,b: (N,4) int16 t,z,y,x ; exact-position matching + nearest for displaced
    sa=set(map(tuple,a[:,1:].tolist())); sb=set(map(tuple,b[:,1:].tolist()))
    return len(sa&sb), len(sa-sb), len(sb-sa)

cache={}
rows=[]
t_start=time.time()
for ws in range(T0-1, T1+1):
    cache[ws]=window_det(ws)
print('encode time', round(time.time()-t_start,1),'s for',T1-T0+2,'windows')
tot={'n_base':0,'n_avg':0,'same':0,'only_base':0,'only_avg':0,'n_B':0}
diffs=[]
for t in range(T0,T1+1):
    A=cache[t-1][1]   # backward-context copy (what the pipeline uses)
    B=cache[t][0]     # forward-context copy (discarded)
    avg=(A+B)/2
    d=(A-B).abs().mean().item(); sd=A.std().item()
    pA=peaks(A,t); pB=peaks(B,t); pAv=peaks(avg,t)
    s,ob,oa=match(pA,pAv)
    sAB,_,_=match(pA,pB)
    diffs.append((t,d,sd,len(pA),len(pB),len(pAv),s,ob,oa,sAB))
    tot['n_base']+=len(pA); tot['n_avg']+=len(pAv); tot['same']+=s; tot['only_base']+=ob; tot['only_avg']+=oa; tot['n_B']+=len(pB)
print('t meanabs_diff  std_A  nA nB nAvg  same lostA newAvg  sameAB')
for r in diffs: print('%3d %8.4f %7.4f %5d %5d %5d %5d %4d %4d %5d'%r)
print('TOTALS',tot)
print('rel diff (mean|A-B| / std A):', round(np.mean([r[1] for r in diffs])/np.mean([r[2] for r in diffs]),4))
print('node count change avg vs base: %+.2f%%'%(100*(tot['n_avg']-tot['n_base'])/max(tot['n_base'],1)))
print('positions changed: %.2f%% of base peaks'%(100*tot['only_base']/max(tot['n_base'],1)))
