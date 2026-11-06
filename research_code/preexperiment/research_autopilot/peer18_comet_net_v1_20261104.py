"""Differentiable causal small-model CoMET, one bounded peer18 round.

No detector features, GT, future frames or pair offsets enter forward().
GT-matched centre/utility targets exist only in separate fit/cal loss data.
"""
import torch
from torch import nn
from torch.nn import functional as F

class CoMETNet(nn.Module):
    def __init__(self,variant,q):
        super().__init__();self.variant=variant;self.q=float(q)
        self.affinity=nn.Sequential(nn.Linear(8,12),nn.Tanh(),nn.Linear(12,1))
        self.noise=nn.Sequential(nn.Linear(8,12),nn.Tanh(),nn.Linear(12,1))
        self.process=nn.Sequential(nn.Linear(5,8),nn.Tanh(),nn.Linear(8,2))
        self.utility=nn.Sequential(nn.Linear(47,16),nn.ReLU(),nn.Linear(16,1))
        self.gru=nn.GRUCell(9,12);self.gru_out=nn.Linear(12,2)
        self.dustbin=nn.Parameter(torch.tensor(0.))
        nn.init.zeros_(self.affinity[-1].weight);nn.init.zeros_(self.affinity[-1].bias)
        nn.init.zeros_(self.noise[-1].weight);nn.init.constant_(self.noise[-1].bias,-.31)
        nn.init.zeros_(self.process[-1].weight);nn.init.constant_(self.process[-1].bias,-4.)
        nn.init.zeros_(self.utility[-1].weight);nn.init.zeros_(self.utility[-1].bias)
    def init(self,device):
        return (torch.zeros(2,device=device,dtype=torch.float64),torch.ones(2,device=device,dtype=torch.float64),
                torch.zeros(12,device=device),torch.zeros((),device=device,dtype=torch.float64))
    def forward(self,frame,state):
        x,p,h,noanchor=state;device=x.device;v=self.variant
        n=len(frame['hard_residual']);prior=p+self.q**2
        if v in ('N4','N5','B2'):
            prior=prior+.005*F.softplus(self.process(frame['motion'])).double()
        use_soft=v in ('N2','N4','N5','B2');use_noise=v in ('N3','N4','N5','B2')
        if use_soft and len(frame['remote_xy']) and len(frame['ego_xy']):
            a,b=frame['remote_xy'],frame['ego_xy'];nr,ne=len(a),len(b)
            residual=b[None,:,:]-a[:,None,:];innovation=residual-x.float()
            desc=frame['pair_descriptor'];inp=torch.cat((innovation/3,desc),-1)
            A=-torch.sum(innovation**2,-1)/2+self.affinity(inp).squeeze(-1)
            full=torch.cat((torch.cat((A,self.dustbin.expand(nr,1)),1),self.dustbin.expand(1,ne+1)),0)
            # Balanced dustbin optimal transport, marginals sum to one.
            total=nr+ne;mu=torch.ones(nr+1,device=device)/total;nu=torch.ones(ne+1,device=device)/total
            mu[-1]=ne/total;nu[-1]=nr/total;logp=full
            for _ in range(8):
                logp=logp+mu.log()[:,None]-torch.logsumexp(logp,1,keepdim=True)
                logp=logp+nu.log()[None,:]-torch.logsumexp(logp,0,keepdim=True)
            weight=torch.exp(logp[:nr,:ne])*total
            if use_noise:weight=weight/(.05+F.softplus(self.noise(inp).squeeze(-1)))**2
            else:weight=weight/.36
            mass=weight.sum().double();z=(weight[:,:,None]*residual).sum((0,1)).double()/mass.clamp_min(1e-12)
            R=1/mass.clamp_min(1e-12)+(.01 if use_noise else 0);available=True
        elif n:
            residual=frame['hard_residual'];desc=frame['hard_descriptor']
            if use_noise:
                inp=torch.cat(((residual.float()-x.float())/3,desc),-1)
                weight=1/(.05+F.softplus(self.noise(inp).squeeze(-1)))**2
                mass=weight.sum().double();z=(weight[:,None]*residual).sum(0).double()/mass;R=1/mass+.01
            else:mass=torch.tensor(n/.36,device=device,dtype=torch.float64);z=residual.double().mean(0);R=torch.tensor(.36/n,device=device,dtype=torch.float64)
            available=True
        else:
            mass=torch.zeros((),device=device,dtype=torch.float64);z=torch.zeros(2,device=device,dtype=torch.float64)
            R=torch.ones((),device=device,dtype=torch.float64);available=False
        innovation=z-x;S=prior+R
        if v in ('B1','B2'):
            # Same causal residual/motion evidence, no sequence/location IDs.
            inp=torch.cat((z.float(),torch.log1p(mass).float().reshape(1),prior.float(),frame['motion'][:3],noanchor.float().reshape(1)/20))
            h=self.gru(inp,h);x=3*torch.tanh(self.gru_out(h)).double();p=prior
        elif available:
            K=prior/S;x=x+K*innovation;p=(1-K)*prior
        else:p=prior
        noanchor=torch.zeros_like(noanchor) if available else noanchor+1
        extra=torch.cat((p,innovation,S,mass.reshape(1),noanchor.reshape(1))).float()
        nll=(torch.log(S)+innovation**2/S).sum() if available else x.sum()*0
        return (x,p,h,noanchor),extra,nll
    def logits(self,features,extra,base_logit):
        if self.variant not in ('N4','N5','B2'):return base_logit
        return base_logit+self.utility(torch.cat((features,extra.expand(len(features),-1)),1)).squeeze(-1)

def detach(state):return tuple(x.detach() for x in state)
