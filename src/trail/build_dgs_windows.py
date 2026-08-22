"""Materialize annotated DGS lexical-pair reconstruction windows from OpenPose."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import ijson, numpy as np
import zstandard

def pose(values):
    a=np.asarray(values,dtype=np.float32).reshape(25,3)
    if a[:,2].mean()<.1:return None
    c=(a[2,:2]+a[5,:2])/2;s=max(float(np.linalg.norm(a[2,:2]-a[5,:2])),1e-4)
    out=np.zeros((25,3),np.float32); good=a[:,2]>=.1;out[good,:2]=(a[good,:2]-c)/s
    return out

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--pairs',type=Path,required=True);p.add_argument('--openpose',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--camera',default='a1');a=p.parse_args()
 rows=list(csv.DictReader(a.pairs.open(encoding='utf8'))); wanted=set()
 for r in rows:
  for x,y in [('left_source_start','left_source_end'),('right_source_start','right_source_end'),('target_start','target_end')]: wanted.update(range(int(r[x]),int(r[y])))
 frames={};cam=None;frame=None;people=[];person=None;field=None
 with a.openpose.open('rb') as raw, zstandard.ZstdDecompressor().stream_reader(raw) as f:
  for pre,event,val in ijson.parse(f):
   if pre=='item.camera' and event=='string':cam=val
   elif pre=='item.frames' and event=='map_key':frame=int(val);people=[]
   elif frame is not None and pre==f'item.frames.{frame}.people.item' and event=='start_map':person=[]
   elif person is not None and pre.endswith('pose_keypoints_2d') and event=='start_array':field='body'
   elif person is not None and field and pre.endswith('pose_keypoints_2d.item') and event=='number':person.append(float(val))
   elif field and pre.endswith('pose_keypoints_2d') and event=='end_array':field=None
   elif frame is not None and pre==f'item.frames.{frame}.people.item' and event=='end_map':people.append(person);person=None
   elif frame is not None and pre==f'item.frames.{frame}' and event=='end_map':
    if cam==a.camera and frame in wanted:
     candidates=[pose(x) for x in people if len(x)==75];candidates=[x for x in candidates if x is not None]
     if candidates:frames[frame]=candidates[0]
    frame=None
 classes={h:i for i,h in enumerate(sorted({r['left_h'] for r in rows}|{r['right_h'] for r in rows}))};L=[];R=[];T=[];LD=[];RD=[];kept=[]
 for r in rows:
  try:
   get=lambda x,y:np.stack([frames[i] for i in range(int(r[x]),int(r[y]))])
   L.append(get('left_source_start','left_source_end'));R.append(get('right_source_start','right_source_end'));T.append(get('target_start','target_end'));ld=np.zeros(len(classes),np.float32);rd=ld.copy();ld[classes[r['left_h']]]=1;rd[classes[r['right_h']]]=1;LD.append(ld);RD.append(rd);kept.append(r)
  except KeyError:pass
 a.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.output,left=np.stack(L),right=np.stack(R),target=np.stack(T),left_descriptor=np.stack(LD),right_descriptor=np.stack(RD),duration=np.full(len(L),T[0].shape[0],np.int64));a.output.with_suffix('.json').write_text(json.dumps({'pairs':len(L),'classes':classes,'source':'annotated DGS boundary with disjoint lexical-segment inputs'},indent=2));print(f'Wrote {len(L)} DGS windows')
if __name__=='__main__':main()
