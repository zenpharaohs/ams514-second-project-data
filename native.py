import ctypes as ct,hashlib,subprocess,tempfile,sys
from pathlib import Path
import numpy as np

def pointer(a):return ct.c_void_p(a.ctypes.data)
def library():
    root=Path(__file__).resolve().parent
    files=[root/'native.c',root/'vendor/cb_core.c',*sorted((root/'vendor/include').glob('*.h'))]
    digest=hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()[:16]
    mac=sys.platform=='darwin';path=Path(tempfile.gettempdir())/('ams514_adaptive_'+digest+('.dylib' if mac else '.so'))
    if not path.exists():subprocess.run(['cc','-O3','-std=c99',*(['-dynamiclib'] if mac else ['-shared','-fPIC']),str(root/'native.c'),'-o',str(path),'-lm'],check=True)
    lib=ct.CDLL(str(path));P=ct.c_void_p;I=ct.c_int;D=ct.c_uint64
    for name,args,ret in [('bank_create',[I,I,D],P),('bank_update',[P,P,P,P],I),('bank_reset',[P,I,P,P],I),
                          ('bank_choose',[P,P],I),('bank_destroy',[P],None),('diagnostic_draw',[P,P,I,I,D,P],I)]:
        f=getattr(lib,name);f.argtypes=args;f.restype=ret
    return lib

class Bank:
    def __init__(self,pools,arms,seed):
        self.lib=library();self.pools=pools;self.handle=self.lib.bank_create(pools,arms,seed)
        if not self.handle:raise MemoryError('CB bank')
    def choose(self):
        out=np.empty(self.pools,np.int32)
        if self.lib.bank_choose(self.handle,pointer(out)):raise RuntimeError('CB draw')
        return out
    def update(self,arms,z,nu):
        arrays=[np.ascontiguousarray(x,dtype) for x,dtype in [(arms,np.int32),(z,float),(nu,float)]]
        if self.lib.bank_update(self.handle,*map(pointer,arrays)):raise RuntimeError('CB update')
    def reset(self,pool,z,nu):
        z=np.ascontiguousarray(z,float);nu=np.ascontiguousarray(nu,float)
        if self.lib.bank_reset(self.handle,int(pool),pointer(z),pointer(nu)):raise RuntimeError('CB reset')
    def close(self):
        if self.handle:self.lib.bank_destroy(self.handle);self.handle=None

def draws(z,nu,n=65536,seed=91):
    z=np.ascontiguousarray(z,float);nu=np.ascontiguousarray(nu,float);out=np.empty((n,len(z)))
    if library().diagnostic_draw(pointer(z),pointer(nu),len(z),n,seed,pointer(out)):raise RuntimeError('Diagnostic CB draw')
    choices=np.argmin(out,axis=1)
    return np.bincount(choices,minlength=len(z))/n
