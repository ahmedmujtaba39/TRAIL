"""Create a deterministic train/test split from fixed transition windows."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--train-output',type=Path,required=True);p.add_argument('--test-output',type=Path,required=True);p.add_argument('--train-fraction',type=float,default=.8);a=p.parse_args()
    v=np.load(a.input); n=len(v['left']); cut=int(n*a.train_fraction)
    for out,sl in ((a.train_output,slice(0,cut)),(a.test_output,slice(cut,n))):
        out.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out,**{key:v[key][sl] for key in v.files})
    print(f'train={cut} test={n-cut}')
if __name__=='__main__':main()
