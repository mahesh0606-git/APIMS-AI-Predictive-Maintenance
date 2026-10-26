"""Temporal-attention APIMS reference model (NOT the proposed topology-aware GNN)."""
import torch
from torch import nn
from torch.nn import functional as F

class TemporalBlock(nn.Module):
    def __init__(self, channels=64, dilation=1):
        super().__init__()
        self.dilation=dilation
        self.layers=nn.ModuleList([nn.Conv1d(channels,channels,3,dilation=dilation) for _ in range(2)])
        self.norms=nn.ModuleList([nn.BatchNorm1d(channels) for _ in range(2)])
    def forward(self,x):
        residual=x
        for conv,norm in zip(self.layers,self.norms):
            x=F.gelu(norm(conv(F.pad(x,(2*self.dilation,0)))))
        return F.gelu(x+residual)

class MDFPE(nn.Module):
    """Input (batch,128,10); outputs p30,p90,rul_mean,rul_log_variance."""
    def __init__(self):
        super().__init__()
        self.projection=nn.Linear(10,64)
        self.blocks=nn.Sequential(*(TemporalBlock(64,d) for d in (1,2,4,8)))
        self.attention=nn.MultiheadAttention(64,4,batch_first=True)
        self.layer_norm=nn.LayerNorm(64)
        self.classifier=nn.Sequential(nn.Linear(64,32),nn.GELU(),nn.Dropout(0.2),nn.Linear(32,2))
        self.rul=nn.Linear(64,2)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(128,10):
            raise ValueError('Expected (batch,128,10)')
        h=self.projection(x).transpose(1,2)
        h=self.blocks(h).transpose(1,2)
        a,_=self.attention(h,h,h,need_weights=False)
        g=self.layer_norm(h+a).mean(dim=1)
        logits=self.classifier(g)
        p30=torch.sigmoid(logits[:,0])
        p90=p30+(1-p30)*torch.sigmoid(logits[:,1])
        r=self.rul(g)
        return {'p30':p30,'p90':p90,'rul_mean':F.softplus(r[:,0]),'rul_logvar':r[:,1]}

def predict_mc(model, x, passes=50, seed=None):
    """MC dropout inference; BatchNorm statistics remain frozen."""
    if passes<2: raise ValueError('passes must be >= 2')
    was_training=model.training
    model.eval()
    for m in model.modules():
        if isinstance(m,nn.Dropout): m.train()
    p30=[];p90=[];rul=[]
    try:
        with torch.no_grad():
            with torch.random.fork_rng(devices=[x.device] if x.is_cuda else []):
                if seed is not None: torch.manual_seed(seed)
                for _ in range(passes):
                    o=model(x)
                    std=torch.exp(0.5*o['rul_logvar']).clamp_min(1e-3)
                    sample=(o['rul_mean']+std*torch.randn_like(std)).clamp_min(0)
                    p30.append(o['p30']);p90.append(o['p90']);rul.append(sample)
        samples=torch.stack(rul)
        return {'p30':torch.stack(p30).mean(0),'p90':torch.stack(p90).mean(0),
                'rul_mean':samples.mean(0),'rul_lower':torch.quantile(samples,.025,dim=0),
                'rul_upper':torch.quantile(samples,.975,dim=0)}
    finally:
        model.train(was_training)
