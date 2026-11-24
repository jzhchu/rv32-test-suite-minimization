"""Read-only independent reconstruction of saved experiment evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .benchmark import make_mutants,kill_matrix
from .experiment import ROOT,write_json
from .reporting import read_rows


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=ROOT/"results")
    args,_=p.parse_known_args();out=args.output
    manifest=json.loads((out/"manifest.json").read_text())
    rows=read_rows(out/"raw_runs.csv")
    summaries=read_rows(out/"summary.csv")
    instances=read_rows(out/"instances.csv")
    instance_map={r["scenario_id"]:r for r in instances}
    details=json.loads((out/"solutions.json").read_text())
    checks={"source_hashes":0,"matrix_hashes":0,"arithmetic_matrices_rebuilt":0,
            "selected_suites_checked":0,"optimal_bounds_checked":0,"raw_rows_checked":0,
            "coverage_failures":0,"result_count_consistent":False}
    for relative,digest in manifest["source_sha256"].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()!=digest:
            raise AssertionError(f"Experiment source changed: {relative}")
        checks["source_hashes"]+=1
    matrices={}
    for sid,record in instance_map.items():
        a=np.load(out/"instances"/(sid+".npz"))["matrix"]
        if hashlib.sha256(a.tobytes()).hexdigest()!=record["matrix_sha256"]:
            raise AssertionError(f"Matrix digest mismatch: {sid}")
        matrices[sid]=a;checks["matrix_hashes"]+=1
        if record["family"]!="synthetic":
            data=json.loads((out/"instances"/(sid+".json")).read_text())
            catalog={m.name:m for m in make_mutants(int(record["width"]))}
            tests=[(t["op"],t["a"],t["b"]) for t in data["tests"]]
            mutants=[catalog[m["id"]] for m in data["mutants"]]
            rebuilt=kill_matrix(tests,mutants,int(record["width"]))
            if not np.array_equal(rebuilt,a):raise AssertionError(f"Rebuild mismatch: {sid}")
            checks["arithmetic_matrices_rebuilt"]+=1
    details_map={}
    for d in details:
        sid=d["scenario_id"];a=matrices[sid]
        chosen=d["selected_original_ids"]
        if len(chosen)!=len(set(chosen)) or min(chosen)<0 or max(chosen)>=a.shape[1]:
            raise AssertionError("Invalid selected IDs")
        active=a.any(axis=1)
        if not np.all(a[active][:,chosen].any(axis=1)):
            raise AssertionError(f"Coverage mismatch {sid}/{d['algorithm']}")
        checks["selected_suites_checked"]+=1
        if d["algorithm"]=="milp" and d["info"]["status"]=="optimal":
            if abs(d["info"]["dual_bound"]-len(chosen))>1e-5 or d["info"]["mip_gap"]>1e-7:
                raise AssertionError("Inconsistent optimum/bound")
            checks["optimal_bounds_checked"]+=1
        details_map[(sid,d["algorithm"],d["rep"])]=d
    for row in rows:
        d=details_map[(row["scenario_id"],row["algorithm"],int(row["rep"]))]
        if row["size"]!=len(d["selected_original_ids"]):raise AssertionError("CSV size mismatch")
        if row["status"]!=d["info"]["status"]:raise AssertionError("CSV status mismatch")
        if row["lp_bound"]>row["size"]+1e-6:raise AssertionError("Inconsistent LP bound")
        if row["optimum"] is not None and row["size"]<row["optimum"]:
            raise AssertionError("Heuristic below certified optimum")
        checks["raw_rows_checked"]+=1
    expected=len(instances)*4*manifest["config"]["timing_repetitions"]
    if len(rows)!=expected or len(details)!=expected or len(summaries)!=len(instances)*4:
        raise AssertionError("Incomplete results")
    checks["result_count_consistent"]=True
    checks["matrix_goal_counts"]={family:dict(Counter(int(r["m_detectable"]) for r in instances if r["family"]==family)) for family in ("arithmetic","width_sensitivity","synthetic")}
    checks["status"]="passed"
    checks["note"]="Reconstructed executed matrices and checked observed detection coverage; solver optimality relies on HiGHS status and matching primal/dual objective bounds."
    write_json(out/"verification.json",checks)
    print(json.dumps(checks,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
