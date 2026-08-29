import json,struct,glob,os,math,statistics as st

def hx(h):
    b=bytes.fromhex(h)
    return (struct.unpack('>f',b)[0],int(h,16),32) if len(b)==4 else (struct.unpack('>d',b)[0],int(h,16),64)
def mono(u,bits):
    s=1<<(bits-1); return -(u&(s-1)) if (u&s) else u
def load(p):
    d=json.load(open(p)); out=[];bits=None
    for rh,ih in d["amps"]:
        r,ru,rb=hx(rh); i,iu,_=hx(ih); bits=rb; out.append((complex(r,i),ru,iu))
    return out,bits
NEG=1e-6; EPS=2**-23

def metrics(A,B,bits,same):
    va=[a[0] for a in A]; vb=[b[0] for b in B]
    na=math.sqrt(sum(abs(x)**2 for x in va)); nb=math.sqrt(sum(abs(x)**2 for x in vb))
    ua=[x/na for x in va]; ub=[x/nb for x in vb]          # normalise before comparing
    ov=sum(x.conjugate()*y for x,y in zip(ua,ub)); aov=abs(ov)
    aov=min(1.0,aov)                                      # clamp: |<a|b>|<=1, float can exceed
    F=aov*aov; infid=max(0.0,1.0-F)
    bures=math.acos(max(-1.0,min(1.0,aov)))               # radians
    trace=math.sqrt(max(0.0,1.0-F))
    l2_raw=math.sqrt(sum(abs(x-y)**2 for x,y in zip(ua,ub)))
    l2_al=math.sqrt(max(0.0,2.0-2.0*aov))                 # global-phase-aligned
    p=[abs(x)**2 for x in ua]; q=[abs(y)**2 for y in ub]
    tvd=0.5*sum(abs(pi-qi) for pi,qi in zip(p,q))
    hell=math.sqrt(max(0.0,1.0-sum(math.sqrt(pi*qi) for pi,qi in zip(p,q))))
    dpmax=max(abs(pi-qi) for pi,qi in zip(p,q))
    shots=(1.0/(tvd*tvd)) if tvd>0 else float('inf')
    mx=max(abs(x-y) for x,y in zip(va,vb))
    us=[];near=0;bd=0
    if same:
        for (av,ar,ai),(bv,br,bi) in zip(A,B):
            for xv,yv,xu,yu in ((av.real,bv.real,ar,br),(av.imag,bv.imag,ai,bi)):
                if xu==yu: continue
                bd+=1
                if max(abs(xv),abs(yv))>=NEG: us.append(abs(mono(xu,bits)-mono(yu,bits)))
                else: near+=1
    return dict(infid=infid,bures=bures,trace=trace,l2r=l2_raw,l2a=l2_al,tvd=tvd,
                hell=hell,dpmax=dpmax,shots=shots,mx=mx,
                umax=max(us) if us else 0,umed=st.median(us) if us else 0,
                bd=bd,near=near,n=len(va))

import sys
# Vendored from the CUDA-side analysis (computed there, independently
# recomputed on the Mac before landing — every load-bearing figure agreed;
# residual scatter was 1-ULP python-libm differences at the 1e-16 floor,
# fittingly). Dirs default to the original hardcoded paths; pass
# <cuda_dir> <mac_dir> to override. Output lands next to this script.
DG = sys.argv[1] if len(sys.argv) > 2 else os.path.expanduser("~/biteq-artifacts-dgx")
MC = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/biteq-artifacts-mac")
C=sorted(os.path.basename(f)[:-9] for f in glob.glob(f"{DG}/*.cpu.json"))
CMP=[("CUDA-Exact vs Metal-Exact",lambda c:f"{DG}/{c}.cuda.exact.json",lambda c:f"{MC}/{c}.metal.exact.json",True),
     ("CUDA-Dec vs Metal-Dec",lambda c:f"{DG}/{c}.cuda.decompose.json",lambda c:f"{MC}/{c}.metal.decompose.json",True),
     ("CUDA-Exact vs CUDA-Dec",lambda c:f"{DG}/{c}.cuda.exact.json",lambda c:f"{DG}/{c}.cuda.decompose.json",True),
     ("CUDA-Exact vs CPU-f64",lambda c:f"{DG}/{c}.cuda.exact.json",lambda c:f"{DG}/{c}.cpu.json",False),
     ("CUDA-Dec vs CPU-f64",lambda c:f"{DG}/{c}.cuda.decompose.json",lambda c:f"{DG}/{c}.cpu.json",False),
     ("Metal-Exact vs CPU-f64",lambda c:f"{MC}/{c}.metal.exact.json",lambda c:f"{MC}/{c}.cpu.json",False),
     ("Metal-Dec vs CPU-f64",lambda c:f"{MC}/{c}.metal.decompose.json",lambda c:f"{MC}/{c}.cpu.json",False)]
res={}
for t,l,r,s in CMP:
    rows=[]
    for c in C:
        f,g=l(c),r(c)
        if os.path.exists(f) and os.path.exists(g):
            A,ab=load(f); B,bb=load(g); rows.append((c,metrics(A,B,ab,s and ab==bb)))
    res[t]=rows
json.dump({t:[(c,m) for c,m in rows] for t,rows in res.items()},
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "metrics.json"),"w"),indent=1)
for t,rows in res.items():
    agg={k:max(m[k] for _,m in rows) for k in ("infid","bures","trace","l2a","tvd","hell","dpmax","mx","umax")}
    agg["shots_min"]=min(m["shots"] for _,m in rows)
    print(f"{t:<28} 1-F={agg['infid']:.2e} T={agg['trace']:.2e} theta={agg['bures']:.2e} "
          f"TVD={agg['tvd']:.2e} maxdP={agg['dpmax']:.2e} max|d|={agg['mx']:.2e} ULP={agg['umax']} "
          f"shots={agg['shots_min']:.1e}")
