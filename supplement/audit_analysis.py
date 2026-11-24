"""Post-hoc scientific audit using frozen primary instances and selections.

No primary result is overwritten. Controls share the same reverse-deletion rule.
Unconstrained variants come from the same synthetic catalog, not independent faults.
"""
from pathlib import Path
import argparse
from collections import defaultdict
from fractions import Fraction
import csv,json,sys,hashlib,itertools,math
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
RESULTS=PROJECT/'results'
sys.path.insert(0,str(PROJECT))
from atm.solvers import prune,repair,validate
from atm.reporting import read_rows


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def summary(values):
    return {'median':float(np.median(values)),'min':float(min(values)),'max':float(max(values))}


def main():
    global HERE, RESULTS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=RESULTS,help='Primary result directory (default: archived results/)')
    parser.add_argument('--output',type=Path,default=HERE,help='Supplement output directory')
    args=parser.parse_args()
    RESULTS=args.results.resolve()
    HERE=args.output.resolve()
    HERE.mkdir(parents=True,exist_ok=True)
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    rows=read_rows(RESULTS/'summary.csv')
    detail=json.loads((RESULTS/'solutions.json').read_text())
    selected={(d['scenario_id'],d['algorithm']):d['selected_original_ids'] for d in detail if d['rep']==0}
    fair=[];fair_ids={};certificates=[];vectors={}
    for row in [r for r in rows if r['algorithm']=='milp']:
        sid=row['scenario_id'];data=np.load(RESULTS/'instances'/f'{sid}.npz')
        original=data['matrix'];a=original[data['active_rows']]
        archived=selected[sid,'random']
        # Original method seeds were fixed across timing repetitions.
        assert all(d['selected_original_ids']==archived for d in detail if d['scenario_id']==sid and d['algorithm']=='random')
        pruned=prune(a,archived);assert validate(a,pruned)
        fair_ids[sid]=pruned
        optimum=row['optimum']
        fair.append({'scenario_id':sid,'family':row['family'],'raw_random_size':len(archived),
                     'pruned_random_size':len(pruned),'optimum':optimum,
                     'gap_pct':100*(len(pruned)/optimum-1) if optimum is not None else None,
                     'coverage':1.0,'selected_original_ids':json.dumps(pruned)})
        # Independently solve the equivalent relaxation WITHOUT upper bounds.
        b=a[:,data['representative_columns']].astype(float)
        result=linprog(np.ones(b.shape[1]),A_ub=-csc_matrix(b),b_ub=-np.ones(b.shape[0]),
                       bounds=(0,None),method='highs-ipm')
        assert result.success
        x=result.x;y=-result.ineqlin.marginals
        primal=max(0.0,float(np.max(1-b@x)),float(np.max(-x)))
        dual=max(0.0,float(np.max(b.T@y-1)),float(np.max(-y)))
        gap=abs(float(x.sum()-y.sum()))
        difference=abs(float(result.fun-row['lp_bound']))
        assert max(primal,dual,gap,difference)<1e-7
        certificates.append({'scenario_id':sid,'lp_objective':float(result.fun),
                             'dual_objective':float(y.sum()),'primal_violation':primal,
                             'dual_violation':dual,'absolute_duality_gap':gap,
                             'difference_to_archived_bounded_lp':difference})
        vectors[sid+'_x']=x;vectors[sid+'_y']=y
    write_csv(out/'audit_random_pruned.csv',fair)
    write_csv(out/'audit_lp_certificates.csv',certificates)
    np.savez_compressed(out/'audit_lp_witnesses.npz',**vectors)
    by_family={}
    for family in ('arithmetic','width_sensitivity','synthetic'):
        f=[r for r in fair if r['family']==family];certified=[r for r in f if r['optimum'] is not None]
        by_family[family]={'instances':len(f),'size':summary([r['pruned_random_size'] for r in f]),
                           'proved_references':len(certified),
                           'optimal_matches':sum(r['pruned_random_size']==r['optimum'] for r in certified),
                           'gap':summary([r['gap_pct'] for r in certified])}
    excluded=[]
    for row in [r for r in rows if r['family']=='arithmetic' and r['algorithm']=='milp' and r['mutant_count']==30]:
        sid=row['scenario_id'];full_sid=sid.replace('_m30_','_mall_')
        small=json.loads((RESULTS/'instances'/f'{sid}.json').read_text())
        full=json.loads((RESULTS/'instances'/f'{full_sid}.json').read_text())
        assert small['tests']==full['tests']
        constrained={m['id'] for m in small['mutants']}
        excluded_rows=[i for i,m in enumerate(full['mutants']) if m['id'] not in constrained]
        assert len(excluded_rows)==35
        full_a=np.load(RESULTS/'instances'/f'{full_sid}.npz')['matrix']
        assert full_a[excluded_rows].any(axis=1).all()
        for method in ('greedy','lp_round','milp','random_pruned'):
            chosen=fair_ids[sid] if method=='random_pruned' else selected[sid,method]
            train_a=np.load(RESULTS/'instances'/f'{sid}.npz')['matrix']
            assert validate(train_a,chosen)
            detected=int(full_a[excluded_rows][:,chosen].any(axis=1).sum())
            excluded.append({'scenario_id':sid,'algorithm':method,'selected_size':len(chosen),
                             'constrained_variants':30,'constrained_coverage':1.0,'excluded_variants':35,
                             'excluded_detected':detected,'excluded_retention_pct':100*detected/35,
                             'original_excluded_retention_pct':100.0})
    write_csv(out/'audit_unconstrained_variants.csv',excluded)
    retained={method:{**summary([r['excluded_retention_pct'] for r in excluded if r['algorithm']==method]),
                      'below_full_retention':sum(r['excluded_detected']<35 for r in excluded if r['algorithm']==method)}
              for method in ('greedy','lp_round','milp','random_pruned')}
    # Exact rational primal and dual witnesses for the triangle example.
    triangle=np.array([[1,1,0],[1,0,1],[0,1,1]],dtype=int)
    half=[Fraction(1,2)]*3
    assert all(sum(int(triangle[i,j])*half[j] for j in range(3))==1 for i in range(3))
    assert all(sum(int(triangle[i,j])*half[i] for i in range(3))==1 for j in range(3))
    assert sum(half)==Fraction(3,2)
    # Enumerate all independent coin outcomes for a tiny rounding/repair check.
    alpha=math.log(4);p=alpha/2;expected_size=0.0
    for bits in itertools.product((0,1),repeat=3):
        initial=[i for i,bit in enumerate(bits) if bit]
        weight=np.prod([p if bit else 1-p for bit in bits])
        chosen=prune(triangle,repair(triangle,initial))
        assert validate(triangle,chosen)
        expected_size+=float(weight)*len(chosen)
    assert expected_size<=alpha*1.5+3/4+1e-12
    checks={'status':'passed','fair_postprocessing_checks':len(fair),'excluded_variant_checks':len(excluded),
            'paired_30_vs_65_variant_instances':len(excluded)//4,'lp_primal_dual_checks':len(certificates),
            'max_primal_violation':max(x['primal_violation'] for x in certificates),
            'max_dual_violation':max(x['dual_violation'] for x in certificates),
            'max_absolute_duality_gap':max(x['absolute_duality_gap'] for x in certificates),
            'max_objective_difference':max(x['difference_to_archived_bounded_lp'] for x in certificates),
            'triangle_rational_witness_value':'3/2','triangle_expected_repaired_size':expected_size,
            'fair_baseline_summary':by_family,'unconstrained_variant_summary':retained,
            'primary_files_modified':False,
            'scope':'Retrospective controls on frozen generated instances; excluded variants are correlated members of the same synthetic catalog.'}
    (out/'audit_summary.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
