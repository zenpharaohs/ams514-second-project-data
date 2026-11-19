"""Meaningful checks of information boundaries, accounting, and numerical oracles."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from playback import PrefixPlayback,run_policy,POLICIES
from generate import loss_difference,polynomial

def protocol_checks():
    # A deliberately nonmonotone table exercises best-checkpoint retention.
    loss=np.array([[[9,8,2,4,1,0],[7,6,5,4,3,2]]],float)
    env=PrefixPlayback(loss);o=env.pull(np.array([1]));assert o.counts.tolist()==[[0,1]] and o.losses.tolist()==[[9,6]]
    try:o.losses[0,0]=0;raise AssertionError('Observation is writable')
    except ValueError:pass
    try:env.pull(np.array([2]));raise AssertionError('Bad arm accepted')
    except IndexError:pass
    boundary=np.array([[[1,0,0,0,0,0],[2,1,1,1,1,1]]],float)
    _,choices,_=run_policy(boundary,'cb',991,horizon=5)
    assert np.array_equal(choices[0],np.array([0,1,0,0,0]))
    for _ in range(4):env.pull(np.array([1]))
    try:env.pull(np.array([1]));raise AssertionError('Tape overrun accepted')
    except IndexError:pass
    rng=np.random.default_rng(44);tapes=1+np.cumsum(rng.uniform(-.01,.01,(4,3,81)),axis=2)
    # Poison unqueryable future observations, retaining the entire reachable prefix.
    altered=tapes.copy();altered[:,:,31:]=rng.uniform(1e8,1e10,altered[:,:,31:].shape)
    for policy in POLICIES:
        v,c,n=run_policy(tapes,policy,771,horizon=30)
        vv,cc,nn=run_policy(altered,policy,771,horizon=30)
        assert np.array_equal(c,cc) and np.array_equal(v,vv) and np.array_equal(n,nn)
        held=tapes[:,:,0].min(1);counts=np.zeros((4,3),int)
        for t in range(30):
            arm=c[:,t];counts[np.arange(4),arm]+=1
            held=np.minimum(held,tapes[np.arange(4),arm,counts[np.arange(4),arm]])
            assert np.array_equal(held,v[:,t+1])
        assert np.array_equal(n,counts) and np.all(n.sum(1)==30)
    return {'future_poisoning_all_policies':True,'prefix_and_budget_accounting':True,'read_only_observations':True,'exhaustion_raises':True,'cb_zero_endpoint_ranks_first':True}

def corpus_checks(folder):
    import mpmath as mp
    mp.mp.dps=70;out={}
    for p in sorted(folder.glob('*.npz')):
        d=np.load(p);meta=json.loads(p.with_suffix('.json').read_text());loss=d['loss']
        A=np.array(list(map(float,meta['alpha'])));B=np.array(list(map(float,meta['beta'])))
        ca=[mp.mpf(v) if '/' not in v else mp.mpf(v.split('/')[0])/mp.mpf(v.split('/')[1]) for v in meta['alpha']]
        cb=[mp.mpf(v) if '/' not in v else mp.mpf(v.split('/')[0])/mp.mpf(v.split('/')[1]) for v in meta['beta']]
        # Independent actual objective evaluation from rational coefficients.
        def exact(a,b):
            a,b=mp.mpf(float(a)),mp.mpf(float(b))
            return mp.mpf(meta['C'])+a*a*mp.polyval(ca[::-1],b)-2*a*mp.polyval(cb[::-1],b)
        pts=d['geometry'].reshape(-1,d['geometry'].shape[2],2)
        recorded=loss.reshape(-1,loss.shape[-1])[:,d['geometry_pulls']]
        rng=np.random.default_rng(221);errors=[];relative_changes=[]
        for _ in range(100):
            i=int(rng.integers(len(pts)));j=int(rng.integers(pts.shape[1]));a,b=pts[i,j]
            true=float(exact(a,b));errors.append(abs(true-recorded[i,j])/max(1,meta['C']))
            # Arbitrary small displacements test the divided-difference evaluator.
            new=np.array([a,b])+rng.normal(size=2)*1e-7
            direct=float(exact(*new)-exact(a,b));stable=float(loss_difference(A,B,np.array([a]),np.array([b]),np.array([new[0]]),np.array([new[1]]))[0])
            relative_changes.append(abs(direct-stable)/max(1,abs(direct)))
        assert max(errors)<1e-8
        assert np.all(np.diff(d['objective_evaluations'],axis=2)>=0)
        assert np.all(np.isfinite(loss)) and np.all(loss>=0)
        maximum_increase=float(np.max(np.diff(loss,axis=2)))
        out[p.stem]=dict(shape=list(loss.shape),max_loss_error_over_C=max(errors),max_difference_error_scaled=max(relative_changes),
                         max_upward_roundoff=maximum_increase,high_precision_samples=100)
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',type=Path);p.add_argument('--out',type=Path)
    args=p.parse_args();result={'protocol':protocol_checks()}
    if args.corpus:result['corpus']=corpus_checks(args.corpus)
    print(json.dumps(result,indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
