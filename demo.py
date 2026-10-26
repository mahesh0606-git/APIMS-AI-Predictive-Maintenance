"""Run with: python demo.py — synthetic inputs and UNTRAINED model."""
import numpy as np
import torch
from apims import MDFPE, normalize, predict_mc, Task, optimize

def main():
    torch.manual_seed(42)
    raw=np.tile(np.array([60,2,0,3,0,20,0,5000,12.1,-8],dtype=np.float32),(128,1))
    x=torch.from_numpy(normalize(raw)).unsqueeze(0)
    model=MDFPE()  # random weights: outputs have NO predictive validity
    result=predict_mc(model,x,passes=12,seed=42)
    print('UNTRAINED demonstration (not real failure probabilities):')
    for key,val in result.items(): print(f'{key}: {val.item():.4f}')
    tasks=[Task('example server',.9,2,250),Task('example switch',.6,1,100)]
    for cost,plan in optimize(tasks,generations=15): print('illustrative schedule:',round(cost,2),plan)

if __name__=='__main__': main()
