"""Student-facing prefix interface and allocation rules. No spong dependency.

Policies see an Observation only. Full tapes and instructor labels stay in the
environment/evaluator. This is an API information contract, not a security sandbox.
"""
from dataclasses import dataclass
from pathlib import Path
import ctypes as ct
import hashlib,os,subprocess,sys,tempfile
import numpy as np

@dataclass(frozen=True)
class Observation:
    total_pulls:int
    counts:np.ndarray
    losses:np.ndarray

def readonly(x):
    y=x.copy();y.flags.writeable=False;return y

class PrefixPlayback:
    def __init__(self,loss):
        if loss.ndim!=3 or np.any(~np.isfinite(loss)):raise ValueError('Expected finite [pool,arm,local_pull] losses')
        self.__tapes=loss
        self.__counts=np.zeros(loss.shape[:2],np.int32)
        self.__current=loss[:,:,0].copy();self.__total=0
    def observation(self):
        return Observation(self.__total,readonly(self.__counts),readonly(self.__current))
    def pull(self,arms):
        arms=np.asarray(arms)
        if arms.shape!=(self.__current.shape[0],) or not np.issubdtype(arms.dtype,np.integer):raise ValueError('One integer arm per pool required')
        if np.any(arms<0) or np.any(arms>=self.__current.shape[1]):raise IndexError('Arm outside pool')
        rows=np.arange(len(arms));next_count=self.__counts[rows,arms]+1
        if np.any(next_count>=self.__tapes.shape[2]):raise IndexError('Tape exhausted; future losses cannot be fabricated')
        self.__counts[rows,arms]=next_count
        self.__current[rows,arms]=self.__tapes[rows,arms,next_count];self.__total+=1
        return self.observation()

def transform(loss,scale):
    # No landscape labels enter this transformation; scale is initial median.
    ratio=np.minimum(loss,scale)/np.maximum(loss,scale)
    z=np.where(loss<=scale,ratio/(1+ratio),1/(1+ratio))
    return np.where(loss>0,np.clip(z,np.nextafter(0.,1.),np.nextafter(1.,0.)),0.)

def native_library():
    root=Path(__file__).resolve().parent
    sources=[root/'cb_bridge.c',root/'vendor/cb_core.c',*sorted((root/'vendor/include').glob('*.h'))]
    h=hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:16]
    suffix='.dylib' if sys.platform=='darwin' else '.so'
    folder=Path(tempfile.gettempdir())/'ams514_cb';folder.mkdir(exist_ok=True)
    library=folder/(h+suffix)
    if not library.exists():
        flags=['-dynamiclib'] if sys.platform=='darwin' else ['-shared','-fPIC']
        subprocess.run([os.environ.get('CC','cc'),'-O3','-std=c99',*flags,str(root/'cb_bridge.c'),'-o',str(library),'-lm'],check=True)
    lib=ct.CDLL(str(library));P=ct.c_void_p
    lib.project_create.argtypes=[ct.c_int,ct.c_int,ct.c_uint64];lib.project_create.restype=P
    lib.project_update.argtypes=[P,P,P,P];lib.project_update.restype=ct.c_int
    lib.project_choose.argtypes=[P,P];lib.project_choose.restype=ct.c_int
    lib.project_destroy.argtypes=[P];lib.project_destroy.restype=None
    return lib

POLICIES=['round_robin','descent_ucb','cb','cb_floor']
class Policy:
    def __init__(self,name,initial,seed=514):
        if name not in POLICIES:raise ValueError(name)
        self.name=name;self.pools,self.arms=initial.losses.shape
        # Initial losses are common information, and no future loss is consulted.
        self.scale=np.nanmedian(np.where(initial.losses>0,initial.losses,np.nan),axis=1)
        self.scale=np.where(np.isfinite(self.scale)&(self.scale>0),self.scale,1.)
        self.rng=np.random.default_rng(seed);self.lib=None;self.bank=None
        if name.startswith('cb'):
            self.lib=native_library();self.bank=self.lib.project_create(self.pools,self.arms,int(seed))
            if not self.bank:raise MemoryError('CB stream allocation failed')
    def choose(self,o):
        if o.total_pulls<self.arms:return np.full(self.pools,o.total_pulls,np.int32)
        if self.name=='round_robin':return np.full(self.pools,o.total_pulls%self.arms,np.int32)
        if self.name=='descent_ucb':
            t=o.total_pulls+1.;z=transform(o.losses,self.scale[:,None])
            index=z-np.sqrt(2*np.log1p(t*np.log(t)**2)/(o.counts+1))
            return np.argmin(index,axis=1).astype(np.int32)
        choices=np.empty(self.pools,np.int32)
        if self.lib.project_choose(self.bank,ct.c_void_p(choices.ctypes.data))!=0:raise RuntimeError('CB draw failed')
        if self.name=='cb_floor':
            explore=self.rng.random(self.pools)<.1
            choices[explore]=self.rng.integers(0,self.arms,int(explore.sum()))
        return choices
    def observe(self,o,arms):
        if self.lib is None:return
        rows=np.arange(self.pools);a=np.ascontiguousarray(arms,np.int32)
        n=np.ascontiguousarray(o.counts[rows,a],np.int32)
        z=np.ascontiguousarray(transform(o.losses[rows,a],self.scale),np.float64)
        if self.lib.project_update(self.bank,*[ct.c_void_p(x.ctypes.data) for x in (a,z,n)])!=0:raise RuntimeError('CB stats replacement failed')
    def close(self):
        if self.bank:self.lib.project_destroy(self.bank);self.bank=None
    def __del__(self):self.close()

def run_policy(loss,name,seed,horizon=None):
    """Return revealed outcomes and decisions; no oracle is evaluated here."""
    horizon=loss.shape[2]-1 if horizon is None else horizon
    env=PrefixPlayback(loss);o=env.observation();policy=Policy(name,o,seed)
    m,k=loss.shape[:2];best=o.losses.min(axis=1);values=np.empty((m,horizon+1));values[:,0]=best
    choices=np.empty((m,horizon),np.uint16)
    try:
        for t in range(1,horizon+1):
            arms=policy.choose(o);o=env.pull(arms);policy.observe(o,arms)
            choices[:,t-1]=arms;best=np.minimum(best,o.losses[np.arange(m),arms]);values[:,t]=best
    finally:policy.close()
    return values,choices,o.counts
