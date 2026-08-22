"""Build DGS Model-T windows directly from annotated video frames."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--pairs',type=Path,required=True);p.add_argument('--video',type=Path,required=True);p.add_argument('--pose-model',type=Path,required=True);p.add_argument('--hand-model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import mediapipe as mp
 from mediapipe.tasks import python
 from mediapipe.tasks.python import vision
 rows=list(csv.DictReader(a.pairs.open(encoding='utf8')));wanted=set()
 for r in rows:
  for x,y in [('left_source_start','left_source_end'),('right_source_start','right_source_end'),('target_start','target_end')]:wanted.update(range(int(r[x]),int(r[y])))
 cap=cv2.VideoCapture(str(a.video));frames={};i=0
 po=vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(a.pose_model)),running_mode=vision.RunningMode.IMAGE,num_poses=1)
 ho=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(a.hand_model)),running_mode=vision.RunningMode.IMAGE,num_hands=2)
 with vision.PoseLandmarker.create_from_options(po) as pd, vision.HandLandmarker.create_from_options(ho) as hd:
  while True:
   ok,bgr=cap.read()
   if not ok:break
   if i in wanted:
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);im=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb);pr=pd.detect(im)
    if pr.pose_landmarks:
     body=np.asarray([[q.x,q.y,q.z] for q in pr.pose_landmarks[0]],np.float32);hr=hd.detect(im);hands=np.zeros((2,21,3),np.float32)
     for pts,side in zip(hr.hand_landmarks,hr.handedness):hands[0 if side[0].category_name.lower()=='left' else 1]=np.asarray([[q.x,q.y,q.z] for q in pts],np.float32)
     frames[i]=np.concatenate([body,hands.reshape(42,3)])
   i+=1
 cap.release();cls={h:n for n,h in enumerate(sorted({r['left_h'] for r in rows}|{r['right_h'] for r in rows}))};L=[];R=[];T=[];LD=[];RD=[]
 for r in rows:
  try:
   get=lambda x,y:np.stack([frames[j] for j in range(int(r[x]),int(r[y]))]);L.append(get('left_source_start','left_source_end'));R.append(get('right_source_start','right_source_end'));T.append(get('target_start','target_end'));x=np.zeros(len(cls),np.float32);y=x.copy();x[cls[r['left_h']]]=1;y[cls[r['right_h']]]=1;LD.append(x);RD.append(y)
  except KeyError:pass
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,left=np.stack(L),right=np.stack(R),target=np.stack(T),left_descriptor=np.stack(LD),right_descriptor=np.stack(RD),duration=np.full(len(L),8,np.int64));a.output.with_suffix('.json').write_text(json.dumps({'pairs':len(L),'classes':cls},indent=2));print(len(L))
if __name__=='__main__':main()
