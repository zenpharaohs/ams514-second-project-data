"""Prefix-only adaptive CB with audited concentration recalibration."""
import time
import numpy as np
from playback import PrefixPlayback,transform
from native import Bank
from calibration import Probabilities,match

class AdaptiveCB:
    def __init__(self,observation,seed,mode='matched',exploration=0.):
        self.mode=mode;self.exploration=exploration
        self.pools,self.arms=observation.losses.shape
        values=np.where(observation.losses>0,observation.losses,np.nan)
        self.scale=np.nanmedian(values,axis=1);self.scale=np.where(np.isfinite(self.scale)&(self.scale>0),self.scale,1.)
        self.nu=np.zeros_like(observation.losses)
        self.below=observation.losses<self.scale[:,None];self.below_count=self.below.sum(1)
        self.threshold=int(np.ceil(.75*self.arms));self.pending=np.zeros(self.pools,bool)
        self.bank=Bank(self.pools,self.arms,seed);self.rng=np.random.default_rng(seed)
        self.prob=Probabilities();self.events=[];self.skipped_zero=0
    def choose(self,o):
        if o.total_pulls<self.arms:return np.full(self.pools,o.total_pulls,np.int32)
        a=self.bank.choose()
        if self.exploration:
            explore=self.rng.random(self.pools)<self.exploration
            a[explore]=self.rng.integers(0,self.arms,int(explore.sum()))
        return a
    def observe(self,o,arms):
        rows=np.arange(self.pools);self.nu[rows,arms]+=1
        losses=o.losses[rows,arms]
        before=self.below[rows,arms].copy();after=losses<self.scale
        self.below[rows,arms]=after;self.below_count+=after.astype(int)-before.astype(int)
        crossed=after&~before
        trigger=crossed&(self.below_count>=self.threshold)
        self.pending|=trigger
        self.bank.update(arms,transform(losses,self.scale),self.nu[rows,arms])
        if self.mode=='fixed' or o.total_pulls<self.arms:return
        if o.total_pulls==self.arms:trigger|=self.pending
        zero=np.any(o.losses==0,axis=1);self.skipped_zero+=int((zero&trigger).sum())
        for pool in np.flatnonzero(trigger&~zero):
            self.recalibrate(pool,o,float(np.median(o.losses[pool])))
        self.pending[:]=False
    def recalibrate(self,pool,o,proposal):
        old_scale=self.scale[pool];loss=o.losses[pool];oldnu=self.nu[pool].copy()
        zold=transform(loss,old_scale);log_ratio=np.log(proposal/old_scale)
        if log_ratio==0:return
        event=dict(pool=int(pool),time=int(o.total_pulls),old_scale=float(old_scale),proposal=float(proposal),
                   zold=zold.copy(),oldnu=oldnu.copy(),accepted=False,backtracks=0,evaluations=0,
                   below_count=int(self.below_count[pool]))
        if self.mode=='naive':
            chosen=proposal;znew=transform(loss,chosen);newnu=oldnu
            event.update(tv=np.nan,max_error=np.nan,accepted=True,kind='naive',audit_resolution=0)
        else:
            target,_=self.prob(zold,oldnu);audit_old=None
            success=False
            for step in range(5):
                chosen=old_scale*np.exp(log_ratio/(2**step));znew=transform(loss,chosen)
                # A partial change must also rearm the count-crossing trigger.
                if np.sum(loss<chosen)>=self.threshold:continue
                # Necessary bound for means on [0,1]: p_i <= 1-(z_i-min z).
                if np.any(target>1-znew+znew.min()+2e-4):continue
                naive,_=self.prob(znew,oldnu)
                if .5*abs(naive-target).sum()<2e-6:
                    newnu=oldnu;info=dict(evaluations=0,kind='already_close')
                else:
                    newnu,info=match(self.prob,zold,oldnu,znew,target=target)
                    info['kind']='fitted'
                event['evaluations']+=info['evaluations']
                if audit_old is None:audit_old,_=self.prob(zold,oldnu,257)
                audit_new,_=self.prob(znew,newnu,257)
                error=abs(audit_new-audit_old);tv=.5*error.sum()
                relevant=audit_old>=1e-4
                relative=np.max(error[relevant]/audit_old[relevant]) if relevant.any() else 0.
                if tv<=2e-4 and relative<=.05:
                    event.update(tv=float(tv),max_error=float(error.max()),relative_error=float(relative),
                                 accepted=True,kind=info['kind'],backtracks=step,audit_resolution=257,
                                 before_prob=audit_old.copy(),after_prob=audit_new.copy())
                    success=True;break
            if not success:
                event.update(tv=float(tv) if 'tv' in locals() else np.inf,max_error=float(error.max()) if 'error' in locals() else np.inf,
                             kind='rejected',new_scale=float(old_scale),znew=zold.copy(),newnu=oldnu.copy())
                self.events.append(event);return
        self.scale[pool]=chosen;self.nu[pool]=newnu;self.bank.reset(pool,znew,newnu)
        self.below[pool]=loss<chosen;self.below_count[pool]=self.below[pool].sum()
        event.update(new_scale=float(chosen),znew=znew.copy(),newnu=newnu.copy())
        self.events.append(event)
    def close(self):self.bank.close()

def run(loss,seed,mode='matched',exploration=0.,horizon=None,progress=None):
    horizon=loss.shape[-1]-1 if horizon is None else horizon
    env=PrefixPlayback(loss);o=env.observation();policy=AdaptiveCB(o,seed,mode,exploration)
    m,k=loss.shape[:2];values=np.empty((m,horizon+1));values[:,0]=o.losses.min(1)
    choices=np.empty((m,horizon),np.uint16);best=values[:,0].copy();start=time.perf_counter()
    try:
        for t in range(1,horizon+1):
            arms=policy.choose(o);o=env.pull(arms);policy.observe(o,arms)
            choices[:,t-1]=arms;best=np.minimum(best,o.losses[np.arange(m),arms]);values[:,t]=best
            if progress and t%128==0:progress(dict(t=t,events=len(policy.events),elapsed=time.perf_counter()-start))
        return values,choices,o.counts,policy.events,policy.scale.copy(),policy.nu.copy()
    finally:policy.close()
