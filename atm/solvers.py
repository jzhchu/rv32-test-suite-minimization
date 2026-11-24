"""Three algorithms for the SAME unit-cost set-cover objective.

LP and MILP use SciPy/HiGHS. Greedy, rounding, repair, and reverse deletion are
implemented here. Timings include all algorithm-specific work, including LP
rounding trials and repairs. Common matrix compression is timed separately.
"""
from dataclasses import dataclass, field
from time import perf_counter
import math
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import csc_matrix


@dataclass
class Solution:
    selected: list[int]
    seconds: float
    info: dict = field(default_factory=dict)
    trace: list[dict] = field(default_factory=list)


def compress(a):
    """Remove zero columns and duplicate kill signatures, preserving one test.

    Only detectable rows enter the optimization. Uncovered original rows are
    recorded by the caller; they are NOT labelled equivalent mutants.
    """
    active=np.flatnonzero(a.any(axis=1))
    b=a[active]
    if len(active)==0: return b[:,:0],np.array([],dtype=int),active
    seen={}; columns=[]
    for j in range(b.shape[1]):
        signature=np.packbits(b[:,j]).tobytes()
        if b[:,j].any() and signature not in seen:
            seen[signature]=j; columns.append(j)
    cols=np.asarray(columns,dtype=int)
    return b[:,cols],cols,active


def validate(a,selected):
    return len(set(selected))==len(selected) and (a.shape[0]==0 or (bool(selected) and bool(np.all(a[:,selected].sum(axis=1)>=1))))


def prune(a,selected):
    """Reverse-delete redundant tests. Used identically for greedy and LP."""
    chosen=list(dict.fromkeys(map(int,selected)))
    if not chosen: return []
    counts=a[:,chosen].sum(axis=1).astype(np.int64)
    removed=set()
    for j in reversed(chosen):
        rows=a[:,j].astype(bool)
        if np.all(counts[rows]>=2):
            counts-=a[:,j]; removed.add(j)
    return [j for j in chosen if j not in removed]


def repair(a,selected,trace=None):
    chosen=list(dict.fromkeys(map(int,selected)))
    covered=a[:,chosen].any(axis=1) if chosen else np.zeros(a.shape[0],dtype=bool)
    while not covered.all():
        gains=a[~covered,:].sum(axis=0)
        # Stable tie break: first column in the original candidate ordering.
        j=int(np.argmax(gains))
        if gains[j]==0: raise ValueError("Uncoverable row")
        chosen.append(j);covered|=a[:,j].astype(bool)
        if trace is not None:
            trace.append({"selected":len(chosen),"covered":int(covered.sum())})
    return chosen


def greedy(a):
    start=perf_counter();trace=[]
    selected=prune(a,repair(a,[],trace))
    return Solution(selected,perf_counter()-start,{"status":"feasible","pruned_size":len(selected)},trace)


def lp_round(a,seed=0,trials=16):
    """Optimal LP + logarithmic randomized rounding + greedy repair.

    Trial 0 uses only (near-)one LP entries then repairs; the other trials use
    p_j=min(1,log(m+1)*x_j). Best feasible cardinality wins. The extra trials
    are an explicit heuristic budget, never hidden as a single LP call.
    """
    start=perf_counter()
    m,n=a.shape
    lp=linprog(np.ones(n),A_ub=-csc_matrix(a,dtype=float),b_ub=-np.ones(m),
               bounds=(0,1),method="highs-ipm",options={"presolve":True})
    if not lp.success: raise RuntimeError(f"LP failed: {lp.message}")
    x=np.clip(lp.x,0,1)
    rng=np.random.default_rng(seed)
    p=np.minimum(1,math.log(m+1)*x)
    best=None;trace=[];best_raw=None;best_repairs=None
    integral=bool(np.all((x<1e-7)|(x>1-1e-7)))
    count=1 if integral else trials
    for trial in range(count):
        initial=np.flatnonzero(x>1-1e-7) if trial==0 else np.flatnonzero(rng.random(n)<p)
        raw=len(initial)
        repaired=repair(a,initial)
        n_repaired=len(repaired)-raw
        candidate=prune(a,repaired)
        if best is None or len(candidate)<len(best):
            best=candidate;best_raw=raw;best_repairs=n_repaired
        trace.append({"trial":trial+1,"incumbent":len(best),"lp_bound":float(lp.fun)})
    return Solution(best,perf_counter()-start,{
        "status":"feasible","lp_bound":float(lp.fun),"lp_iterations":int(lp.nit),
        "lp_fractional_variables":int(np.sum((x>1e-7)&(x<1-1e-7))),
        "rounding_trials":count,"winning_initial_size":best_raw,"winning_repairs":best_repairs},trace)


def integer_program(a,time_limit=5.0):
    start=perf_counter();m,n=a.shape
    result=milp(np.ones(n),integrality=np.ones(n),bounds=Bounds(0,1),
                constraints=LinearConstraint(csc_matrix(a,dtype=float),np.ones(m),np.full(m,np.inf)),
                options={"presolve":True,"time_limit":time_limit,"mip_rel_gap":0.0})
    if result.x is None:
        # No incumbent: report explicitly and use a feasible fallback, not an optimum.
        chosen=prune(a,repair(a,[]))
        fallback=True
    else:
        chosen=np.flatnonzero(result.x>0.5).tolist();fallback=False
    return Solution(chosen,perf_counter()-start,{
        "status":"optimal" if result.status==0 else "limit_or_failure",
        "solver_status":int(result.status),"message":str(result.message),
        "mip_gap":float(result.mip_gap) if getattr(result,"mip_gap",None) is not None else None,
        "dual_bound":float(result.mip_dual_bound) if getattr(result,"mip_dual_bound",None) is not None else None,
        "nodes":int(result.mip_node_count) if getattr(result,"mip_node_count",None) is not None else None,
        "fallback":fallback})


def random_order(a,seed=0):
    start=perf_counter();rng=np.random.default_rng(seed)
    covered=np.zeros(a.shape[0],dtype=bool);chosen=[]
    for j in rng.permutation(a.shape[1]):
        chosen.append(int(j));covered|=a[:,j].astype(bool)
        if covered.all():break
    return Solution(chosen,perf_counter()-start,{"status":"feasible"})
