"""Concentration matching within the CB conjugate family, with fixed new modes."""
from pathlib import Path
import ctypes as ct,hashlib,subprocess,tempfile,sys
import numpy as np
from scipy.optimize import least_squares

class Probabilities:
    def __init__(self,nodes=129):
        root=Path(__file__).resolve().parent;src=root/'cb_probabilities.c'
        digest=hashlib.sha256(src.read_bytes()).hexdigest()[:16]
        mac=sys.platform=='darwin';lib=Path(tempfile.gettempdir())/('ams514_prob_'+digest+('.dylib' if mac else '.so'))
        if not lib.exists():subprocess.run(['cc','-O3','-std=c99',*(['-dynamiclib'] if mac else ['-shared','-fPIC']),str(src),'-o',str(lib),'-lm'],check=True)
        self.lib=ct.CDLL(str(lib));self.lib.cb_winner.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_int,ct.c_int,ct.c_void_p,ct.c_void_p]
        self.lib.cb_winner.restype=ct.c_int;self.nodes=nodes
    def __call__(self,z,nu,nodes=None):
        z=np.ascontiguousarray(z,np.float64);nu=np.ascontiguousarray(nu,np.float64);k=len(z)
        p=np.empty(k);jac=np.empty((k,k))
        rc=self.lib.cb_winner(ct.c_void_p(z.ctypes.data),ct.c_void_p(nu.ctypes.data),k,nodes or self.nodes,ct.c_void_p(p.ctypes.data),ct.c_void_p(jac.ctypes.data))
        if rc:raise RuntimeError(f'Probability quadrature error {rc}')
        return p,jac

def match(prob,z_old,nu_old,z_new,max_evals=30,target=None):
    if target is None:target,_=prob(z_old,nu_old)
    if np.any(z_new==0):return nu_old.copy(),dict(tv=0.,max_error=0.,evaluations=0,success=True)
    initial=np.log(np.clip(nu_old,.02,1e6));cache={}
    floor=1e-8
    def evaluate(x):
        if 'x' not in cache or not np.array_equal(x,cache['x']):
            p,j=prob(z_new,np.exp(x));cache.update(x=x.copy(),p=p,j=j)
        return cache['p'],cache['j']
    def fun(x):
        p,j=evaluate(x)
        return np.r_[np.log(p+floor)-np.log(target+floor),1e-6*(x-initial)]
    def jac(x):
        p,j=evaluate(x)
        return np.r_[j/(p[:,None]+floor),1e-6*np.eye(len(x))]
    result=least_squares(fun,initial,jac=jac,bounds=(np.log(.02),np.log(1e6)),max_nfev=max_evals,
                         xtol=1e-7,ftol=1e-7,gtol=1e-7)
    fitted=np.exp(result.x);p,_=prob(z_new,fitted)
    error=abs(p-target);tv=.5*error.sum()
    return fitted,dict(tv=float(tv),max_error=float(error.max()),evaluations=result.nfev,
                       success=bool(tv<5e-5),target=target.tolist(),after=p.tolist())
