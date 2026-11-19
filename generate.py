"""Generate indexed population-loss tapes. No allocator is used in generation."""
import argparse,json,sys,time,subprocess
from pathlib import Path
import numpy as np

CASES=['quadratic_gd','near_saddle_gd','plateau_gd','plateau_ngd']

def polynomial(c,x):
    return np.polynomial.polynomial.polyval(x,c)

def differential(c):
    return np.arange(1,len(c))*np.asarray(c[1:])

def poly_difference(c,old,new):
    """Horner divided difference; avoids subtracting nearly equal values."""
    value=np.full_like(old,c[-1]);div=np.zeros_like(old)
    for coefficient in c[-2::-1]:
        div=div*new+value;value=value*old+coefficient
    return (new-old)*div

def loss_difference(A,B,a,b,ca,cb):
    da=ca-a;aa=polynomial(A,b);bb=polynomial(B,b)
    return da*(2*(a*aa-bb)+da*aa)+ca*ca*poly_difference(A,b,cb)-2*ca*poly_difference(B,b,cb)

def coefficients(m):
    A=np.array(list(map(float,m.alpha)));B=np.array(list(map(float,m.beta)))
    D=np.zeros(2*len(m.g)-3)
    for k in range(1,len(m.g)):
        for l in range(1,len(m.g)):
            D[k+l-2]+=k*l*float(m.g[k]*m.g[l]*m.mu[k+l])
    return A,B,D,float(m.C)

def generate(name,pools,arms,horizon,chunk,seed,spong):
    sys.path.insert(0,str(spong/'src'))
    from spong import model,zoo,sturm
    if name.startswith('plateau'):
        case=zoo.get('normal-symmetric-small-integer-d9')
        m=model.build(case.f,case.g,zoo.moments_for(case,19))
        bounds=np.array([[-50.,-.6],[50.,.6]])
        optimum=float(m.C-m.beta[-1]**2/m.alpha[-1]);target=optimum+.01*(float(m.C)-optimum)
    else:
        m=model.build((1,1,1),(1,1,1),model.moments_normal01(5))
        bounds=np.array([[-2.,-2.],[2.,2.]])
        optimum=0.;target=1e-4
    A,B,D,C=coefficients(m);Ap=differential(A);Bp=differential(B)
    enumeration=sturm.enumerate_critical_points(m)
    root_points=np.array([[p.a,p.b] for p in enumeration.points])
    rng=np.random.default_rng(np.random.SeedSequence([seed,CASES.index(name)//2 if name.startswith('plateau') else CASES.index(name)]))
    start=rng.uniform(bounds[0],bounds[1],(pools*arms,2))
    law={'kind':'iid uniform rectangle','bounds':bounds.tolist()}
    if name=='near_saddle_gd':
        saddle=next(p for p in enumeration.points if p.kind=='saddle')
        center=np.array([saddle.a,saddle.b]);e,v=np.linalg.eigh(m.hessL(*center))
        sign=rng.choice([-1.,1.],len(start));radius=10.**rng.uniform(-11,-2,len(start))
        start=center+sign[:,None]*radius[:,None]*v[:,0]
        law={'kind':'iid near-saddle unstable-eigenvector offsets','center':center.tolist(),
             'direction':v[:,0].tolist(),'log10_radius_uniform':[-11,-2],'sign_probabilities':[.5,.5]}
    z=start.copy();n=len(z);trial=np.full(n,.05 if name.endswith('_gd') else 1.)
    status=np.zeros(n,dtype=np.uint8);stop_at=np.full(n,horizon+1,dtype=np.int32)
    loss=np.empty((n,horizon+1));loss[:,0]=m.L(*z.T)
    evaluations=np.zeros((n,horizon+1),np.uint32);gradients=np.zeros_like(evaluations)
    geom_steps=np.unique(np.r_[0,np.arange(16,horizon+1,16),horizon]);geom=np.empty((n,len(geom_steps),2));geom[:,0]=z
    damping=.001;armijo=1e-4;maximum=.05 if name.endswith('_gd') else 1.
    for pull in range(1,horizon+1):
        ev=np.zeros(n,np.uint32);gv=np.zeros(n,np.uint32)
        for _ in range(chunk):
            idx=np.flatnonzero(status==0)
            if not len(idx):break
            a,b=z[idx].T;aa=polynomial(A,b);bb=polynomial(B,b);ap=polynomial(Ap,b);bp=polynomial(Bp,b)
            q=a*(a*aa-2*bb);g=np.column_stack([2*(a*aa-bb),a*(a*ap-2*bp)])
            gv[idx]+=1;ev[idx]+=1
            if name.endswith('_gd'):direction=-g
            else:
                diag1=np.maximum(aa,1e-100);diag2=np.maximum(a*a*polynomial(D,b),1e-100)
                s1,s2=np.sqrt(diag1),np.sqrt(diag2);rho=.5*a*ap/(s1*s2)
                rhs1,rhs2=-g[:,0]/s1,-g[:,1]/s2
                den=(1+damping)**2-rho*rho
                direction=np.column_stack([((1+damping)*rhs1-rho*rhs2)/den/s1,
                                           ((1+damping)*rhs2-rho*rhs1)/den/s2])
            slope=np.sum(g*direction,axis=1);step=trial[idx].copy();accepted=np.zeros(len(idx),bool)
            finite=np.all(np.isfinite(direction),axis=1)&np.isfinite(slope)
            for _half in range(48):
                j=np.flatnonzero(~accepted&finite)
                if not len(j):break
                candidate=z[idx[j]]+step[j,None]*direction[j]
                ca,cb=candidate.T
                with np.errstate(over='ignore',invalid='ignore'):
                    change=loss_difference(A,B,a[j],b[j],ca,cb)
                ev[idx[j]]+=1
                actual_slope=np.sum(g[j]*(candidate-z[idx[j]]),axis=1)
                ok=np.isfinite(change)&(change<0)&(actual_slope<0)&(change<=armijo*actual_slope)
                good=j[ok];z[idx[good]]=candidate[ok];accepted[good]=True
                step[j[~ok]]*=.5
            failed=~accepted
            status[idx[failed]]=np.where(finite[failed],1,2)
            stop_at[idx[failed]]=pull
            trial[idx[accepted]]=np.minimum(maximum,2*step[accepted])
        with np.errstate(over='ignore',invalid='ignore'):
            value=np.asarray(m.L(*z.T),float)
        if np.any(~np.isfinite(value)):raise FloatingPointError('Nonfinite accepted loss')
        if np.any(value< -1e-12*max(1,C)):raise FloatingPointError('Negative squared loss')
        loss[:,pull]=np.maximum(value,0.)
        evaluations[:,pull]=evaluations[:,pull-1]+ev+1
        gradients[:,pull]=gradients[:,pull-1]+gv
        if pull in geom_steps:geom[:,int(np.searchsorted(geom_steps,pull))]=z
        if pull%512==0:print(name,'pull',pull,'active',int((status==0).sum()),'target',int((loss[:,pull]<=target).sum()),flush=True)
    # Tiny evaluation roundoff is retained; monotonicity is checked with a stated tolerance.
    increase=np.diff(loss,axis=1)
    tolerance=1e-11*np.maximum(1,np.max(loss,axis=1))[:,None]
    if np.any(increase>tolerance):raise AssertionError('Unexpected upward accepted trajectory')
    metadata=dict(name=name,pools=pools,arms=arms,horizon=horizon,chunk_steps=chunk,seed=seed,
        objective='E[(f(X)-a*g(b*X))^2]',moment_law='normal01',f=list(map(str,m.f)),g=list(map(str,m.g)),
        alpha=list(map(str,m.alpha)),beta=list(map(str,m.beta)),C=C,optimum=optimum,target_loss=target,
        initialization=law,optimizer='population GD with Armijo backtracking' if name.endswith('_gd') else 'population damped Fisher/Gauss-Newton with Armijo backtracking',
        maximum_step=maximum,damping=damping if name.endswith('ngd') else None,armijo=armijo,
        max_halvings=48,termination={'0':'recording cap; no endpoint assertion','1':'line-search precision limit; absorbing last finite state','2':'nonfinite direction; absorbing last finite state'},
        status_counts={str(v):int((status==v).sum()) for v in range(3)},known_critical_points=root_points.tolist(),
        normalization_scale=C-optimum,notes=['Population-moment gradients; no minibatch randomness.',
        'Optimizer parameters fixed before final independent-pool generation.',
        'Damped NGD is a declared variant, not the exact-rational spong demo implementation.',
        'Armijo uses Horner divided differences for loss changes, avoiding subtraction of nearly equal plateau losses.',
        'Stopped tapes retain loss; subsequent pulls spend allocation slots and loss-evaluation cost, but no gradient work.',
        'Geometric snapshots are instructor diagnostics and never passed to allocators.'])
    arrays=dict(loss=loss.reshape(pools,arms,horizon+1),initial_parameters=start.reshape(pools,arms,2),
        objective_evaluations=evaluations.reshape(pools,arms,horizon+1),gradient_evaluations=gradients.reshape(pools,arms,horizon+1),
        terminal_status=status.reshape(pools,arms),stop_at=stop_at.reshape(pools,arms),
        geometry=geom.reshape(pools,arms,len(geom_steps),2),geometry_pulls=geom_steps)
    return arrays,metadata

def main():
    p=argparse.ArgumentParser();p.add_argument('--spong',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'corpus');p.add_argument('--pools',type=int,default=512)
    p.add_argument('--arms',type=int,default=8);p.add_argument('--horizon',type=int,default=2048)
    p.add_argument('--chunk',type=int,default=4);p.add_argument('--seed',type=int,default=20261118)
    p.add_argument('--cases',nargs='+',default=CASES,choices=CASES);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    for name in args.cases:
        t=time.perf_counter();arrays,metadata=generate(name,args.pools,args.arms,args.horizon,args.chunk,args.seed,args.spong)
        metadata['generation_seconds']=time.perf_counter()-t
        metadata['spong_commit']=subprocess.check_output(['git','-C',str(args.spong),'rev-parse','HEAD'],text=True).strip()
        np.savez_compressed(args.out/(name+'.npz'),**arrays)
        (args.out/(name+'.json')).write_text(json.dumps(metadata,indent=2)+'\n')
        print('SAVED',name,metadata['generation_seconds'],metadata['status_counts'],flush=True)
if __name__=='__main__':main()
