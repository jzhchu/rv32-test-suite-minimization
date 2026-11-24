"""Reproducible runner; every selected suite is checked on the original matrix."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import sys
import numpy as np
import scipy
from .benchmark import make_mutants,choose_mutants,generate_tests,kill_matrix,synthetic_matrix,reference
from .solvers import compress,validate,greedy,lp_round,integer_program,random_order

ROOT=Path(__file__).resolve().parents[1]
ALGORITHMS=("greedy","lp_round","milp","random")


def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")


def scenarios(cfg,quick=False):
    seeds=cfg["seeds"][:1] if quick else cfg["seeds"]
    ar=cfg["arithmetic"]
    sizes=ar["candidate_sizes"][:1] if quick else ar["candidate_sizes"]
    for n,ratio,count,seed in itertools.product(sizes,ar["boundary_ratios"],ar["mutant_counts"],seeds):
        yield {"family":"arithmetic","width":32,"n":n,"boundary_ratio":ratio,"mutant_count":count,"seed":seed}
    if not quick:
        w=cfg["width_sensitivity"]
        for width,seed in itertools.product(w["widths"],seeds):
            yield {"family":"width_sensitivity","width":width,"n":w["n"],"boundary_ratio":w["boundary_ratio"],"mutant_count":"all","seed":seed}
    sy=cfg["synthetic_stress"]
    for m,n,density,seed in itertools.product(sy["rows"][:1] if quick else sy["rows"],sy["columns"][:1] if quick else sy["columns"],sy["densities"],seeds):
        yield {"family":"synthetic","m":m,"n":n,"density":density,"seed":seed}


def scenario_id(s):
    if s["family"]=="synthetic": return f"syn_m{s['m']}_n{s['n']}_d{s['density']}_s{s['seed']}"
    return f"arith_w{s['width']}_n{s['n']}_b{s['boundary_ratio']}_m{s['mutant_count']}_s{s['seed']}"


def replay_time(tests,width,target_seconds=0.015):
    """Correct-model regression replay against precomputed expected outputs.

    A calibrated batch amortizes the clock overhead for microsecond-sized suites.
    It is an in-process Python benchmark, not a production-library speedup.
    """
    cases=[(op,a,b,reference(op,a,b,width)) for op,a,b in tests]
    def once():
        checksum=0
        for op,a,b,expected in cases:
            actual=reference(op,a,b,width)
            if actual!=expected: raise AssertionError("Replay failed")
            checksum^=actual
        return checksum
    start=perf_counter();once();duration=perf_counter()-start
    batch=min(20000,max(1,math.ceil(target_seconds/max(duration,1e-8))))
    measured=[]
    for _ in range(3):
        start=perf_counter()
        for _ in range(batch): once()
        measured.append((perf_counter()-start)/batch)
    return float(np.median(measured))


def solve(name,a,cfg,seed):
    if name=="greedy": return greedy(a)
    if name=="lp_round": return lp_round(a,seed,cfg["rounding_trials"])
    if name=="milp": return integer_program(a,cfg["milp_time_limit_seconds"])
    return random_order(a,seed)


def summarize(rows):
    groups=defaultdict(list)
    for row in rows: groups[(row["scenario_id"],row["algorithm"])].append(row)
    summaries=[]
    for (sid,algorithm),group in groups.items():
        first=group[0]
        out={k:v for k,v in first.items() if k not in ("rep","solver_seconds","size","optimality_gap_pct")}
        out.update({"size":float(np.median([r["size"] for r in group])),
                    "solver_ms_median":1000*float(np.median([r["solver_seconds"] for r in group])),
                    "solver_ms_min":1000*min(r["solver_seconds"] for r in group),
                    "solver_ms_max":1000*max(r["solver_seconds"] for r in group),
                    "size_min":min(r["size"] for r in group),"size_max":max(r["size"] for r in group)})
        gaps=[r["optimality_gap_pct"] for r in group if r["optimality_gap_pct"] is not None]
        out["optimality_gap_pct"]=float(np.median(gaps)) if gaps else None
        out["reduction_pct"]=100*(1-out["size"]/out["n_original"])
        out["selection_setup_ms"]=out["preprocess_seconds"]*1000+out["solver_ms_median"]
        out["total_setup_ms"]=1000*(out["generation_seconds"]+out["matrix_seconds"])+out["selection_setup_ms"]
        saved=(out.get("replay_raw_seconds") or 0)-(out.get("replay_selected_seconds") or 0)
        out["break_even_replays_full"]=math.ceil(out["total_setup_ms"]/1000/saved) if saved>0 else None
        out["break_even_replays_cached_matrix"]=math.ceil(out["selection_setup_ms"]/1000/saved) if saved>0 else None
        summaries.append(out)
    return summaries


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"config.json")
    parser.add_argument("--output",type=Path,default=ROOT/"results")
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--skip-replay",action="store_true")
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text());out=args.output.resolve()
    if args.quick: cfg["timing_repetitions"]=1
    out.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/"atm").glob("*.py"))}
    manifest={"start_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,
              "numpy":np.__version__,"scipy":scipy.__version__,"platform":platform.platform(),
              "machine":platform.machine(),"logical_cpus":os.cpu_count(),"config":cfg,
              "config_sha256":hashlib.sha256(args.config.read_bytes()).hexdigest(),"source_sha256":sources,
              "quick":args.quick,"independent_data_seeds":len(cfg["seeds"][:1] if args.quick else cfg["seeds"]),
              "timing_policy":"single-thread BLAS/HiGHS requested via environment; 3 timing reps in rotated order, fixed algorithm seed across reps; warmup excluded",
              "random_policy":"random permutation of the same compressed candidates, no pruning; baseline only"}
    write_json(out/"manifest.json",manifest)
    # Warm up Python and native solvers before measuring.
    warm=np.array([[1,0,1],[0,1,1]],dtype=np.uint8)
    for name in ALGORITHMS: solve(name,warm,cfg,0)
    all_rows=[];instance_rows=[];details=[]
    all_scenarios=list(scenarios(cfg,args.quick));started=perf_counter()
    for index,s in enumerate(all_scenarios):
        sid=scenario_id(s);start=perf_counter()
        tests=None;catalog=None
        if s["family"]!="synthetic":
            catalog=make_mutants(s["width"])
            count=len(catalog) if s["mutant_count"]=="all" else s["mutant_count"]
            ids=choose_mutants(catalog,count,s["seed"])
            muts=[catalog[i] for i in ids]
            tests=generate_tests(s["n"],s["width"],s["boundary_ratio"],s["seed"])
            gen_seconds=perf_counter()-start;start=perf_counter()
            a=kill_matrix(tests,muts,s["width"])
            matrix_seconds=perf_counter()-start
            metadata={"tests":[{"id":j,"op":t[0],"a":t[1],"b":t[2]} for j,t in enumerate(tests)],
                      "mutants":[{"id":m.name,"op":m.op,"description":m.description} for m in muts]}
            write_json(out/"instances"/(sid+".json"),metadata)
        else:
            a=synthetic_matrix(s["m"],s["n"],s["density"],s["seed"])
            gen_seconds=perf_counter()-start;matrix_seconds=0.0
        start=perf_counter();b,columns,active=compress(a);preprocess=perf_counter()-start
        if not len(active): raise RuntimeError(f"No detectable goals: {sid}")
        path=out/"instances"/(sid+".npz");path.parent.mkdir(exist_ok=True)
        np.savez_compressed(path,matrix=a,representative_columns=columns,active_rows=active)
        base={**s,"scenario_id":sid,"n_original":a.shape[1],"m_original":a.shape[0],
              "n_effective":b.shape[1],"m_detectable":b.shape[0],
              "undetected_mutants":int(a.shape[0]-b.shape[0]),"actual_density":float(a.mean()),
              "generation_seconds":gen_seconds,"matrix_seconds":matrix_seconds,"preprocess_seconds":preprocess,
              "matrix_sha256":hashlib.sha256(a.tobytes()).hexdigest()}
        instance_rows.append(base)
        per_name=defaultdict(list)
        for rep in range(cfg["timing_repetitions"]):
            order=list(ALGORITHMS)
            np.random.default_rng(s["seed"]+rep*1009).shuffle(order)
            for name in order:
                solution=solve(name,b,cfg,s["seed"]+100003)
                if not validate(b,solution.selected): raise AssertionError(f"Invalid {sid}/{name}")
                original_ids=columns[solution.selected].tolist()
                if not np.all(a[active][:,original_ids].any(axis=1)):
                    raise AssertionError("Coverage lost on original matrix")
                per_name[name].append(solution)
                details.append({"scenario_id":sid,"algorithm":name,"rep":rep,
                                "selected_original_ids":original_ids,"info":solution.info,"trace":solution.trace})
        optimal_sizes=[len(sol.selected) for sol in per_name["milp"] if sol.info["status"]=="optimal"]
        optimum=min(optimal_sizes) if optimal_sizes else None
        lp_bound=per_name["lp_round"][0].info["lp_bound"]
        if optimum is not None and lp_bound>optimum+1e-6: raise AssertionError("LP bound exceeds optimum")
        raw_replay=replay_time(tests,s["width"]) if tests and not args.skip_replay else None
        for name,solutions in per_name.items():
            selected_tests=[tests[int(columns[j])] for j in solutions[0].selected] if tests else None
            reduced_replay=replay_time(selected_tests,s["width"]) if selected_tests and not args.skip_replay else None
            for rep,sol in enumerate(solutions):
                size=len(sol.selected)
                if optimum is not None and size<optimum: raise AssertionError("Heuristic beats certified optimum")
                all_rows.append({**base,"algorithm":name,"rep":rep,"size":size,
                                 "solver_seconds":sol.seconds,"coverage":1.0,"optimum":optimum,
                                 "optimality_gap_pct":100*(size/optimum-1) if optimum else None,
                                 "lp_bound":lp_bound,"lp_integrality_gap":optimum/lp_bound if optimum else None,
                                 "lp_fractional_variables":per_name["lp_round"][0].info["lp_fractional_variables"],
                                 "status":sol.info["status"],"replay_raw_seconds":raw_replay,
                                 "replay_selected_seconds":reduced_replay})
        write_csv(out/"raw_runs.csv",all_rows)
        write_csv(out/"instances.csv",instance_rows)
        write_json(out/"solutions.json",details)
        print(f"[{index+1:03}/{len(all_scenarios)}] {sid}: goals={b.shape[0]}, columns={b.shape[1]}, "
              f"sizes="+str({k:len(v[0].selected) for k,v in per_name.items()})+
              f", optimum={optimum}, elapsed={perf_counter()-started:.1f}s",flush=True)
    summaries=summarize(all_rows)
    write_csv(out/"summary.csv",summaries)
    manifest["finish_utc"]=datetime.now(timezone.utc).isoformat()
    manifest["elapsed_seconds"]=perf_counter()-started
    manifest["instances"]=len(instance_rows);manifest["algorithm_runs"]=len(all_rows)
    manifest["coverage_checks_passed"]=len(all_rows)
    manifest["milp_optimal_runs"]=sum(r["algorithm"]=="milp" and r["status"]=="optimal" for r in all_rows)
    manifest["undetected_rows_total"]=sum(r["undetected_mutants"] for r in instance_rows if r["family"]!="synthetic")
    write_json(out/"manifest.json",manifest)
    print("FINISHED",json.dumps({k:manifest[k] for k in ("instances","algorithm_runs","elapsed_seconds","milp_optimal_runs")}),flush=True)


if __name__=="__main__": main()
