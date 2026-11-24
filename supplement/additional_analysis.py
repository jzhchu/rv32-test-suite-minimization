"""Recreate figures/tables and the seven-matrix rounding-budget ablation.

Read archived primary results by default, or select a fresh run with --results.
Primary input data and timings are never overwritten by this script.
"""
from pathlib import Path
import argparse
import sys,csv,json,time
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parent
RESULTS=PROJECT/'results'
sys.path.insert(0,str(PROJECT))
from atm.solvers import lp_round,validate
from atm.reporting import read_rows,stats

COLORS={'greedy':'#177E89','lp_round':'#D68400','milp':'#6658A6'}
METHODS=('greedy','lp_round','milp')
NAMES={'greedy':'Greedy','lp_round':'LP rounding','milp':'IP'}


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    global HERE, RESULTS
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=RESULTS,help='Primary result directory (default: archived results/)')
    parser.add_argument('--output',type=Path,default=HERE,help='Supplement output directory')
    parser.add_argument('--reuse-ablation',action='store_true',help='Reuse the output directory’s saved rounding_ablation.csv')
    args=parser.parse_args()
    RESULTS=args.results.resolve()
    HERE=args.output.resolve()
    HERE.mkdir(parents=True,exist_ok=True)
    (HERE/'analysis').mkdir(exist_ok=True)
    (HERE/'figures').mkdir(exist_ok=True)
    rows=read_rows(RESULTS/'summary.csv')
    raw=read_rows(RESULTS/'raw_runs.csv')
    details=json.loads((RESULTS/'solutions.json').read_text())
    # Predeclared slices: seed 11, all three main sizes at ratio .8/all mutants;
    # all four (m,density) combinations at n=640 for the structural supplement.
    ids=[f'arith_w32_n{n}_b0.8_mall_s11' for n in (300,1200,4800)]
    ids += [f'syn_m{m}_n640_d{d}_s11' for m in (40,80) for d in (.05,.15)]
    ab=[];started=time.perf_counter()
    reuse=args.reuse_ablation
    # These are algorithm RNG seeds, NOT additional independently sampled matrices.
    for sid in ([] if reuse else ids):
        data=np.load(RESULTS/'instances'/f'{sid}.npz')
        a=data['matrix'][data['active_rows']][:,data['representative_columns']]
        for budget in (1,4,16,64):
            for seed in range(101,111):
                sol=lp_round(a,seed=seed,trials=budget)
                assert validate(a,sol.selected)
                ab.append({'scenario_id':sid,'family':'arithmetic' if sid.startswith('arith') else 'synthetic',
                           'budget':budget,'rounding_seed':seed,'size':len(sol.selected),
                           'seconds':sol.seconds,'lp_bound':sol.info['lp_bound'],
                           'actual_trials':sol.info['rounding_trials']})
    if reuse:
        ab=read_rows(HERE/'analysis/rounding_ablation.csv')
    else:
        write_csv(HERE/'analysis/rounding_ablation.csv',ab)
    summary=[]
    for family in ('arithmetic','synthetic'):
        for budget in (1,4,16,64):
            sub=[r for r in ab if r['family']==family and r['budget']==budget]
            per=defaultdict(list)
            for r in sub:per[r['scenario_id']].append(r['size'])
            summary.append({'family':family,'budget':budget,'matrices':len(per),'calls':len(sub),
                            'median_size':float(np.median([r['size'] for r in sub])),
                            'min_size':min(r['size'] for r in sub),'max_size':max(r['size'] for r in sub),
                            'median_ms':float(np.median([r['seconds'] for r in sub]))*1000,
                            'varying_matrices':sum(len(set(v))>1 for v in per.values()),
                            'max_within_matrix_range':max(max(v)-min(v) for v in per.values())})
    write_csv(HERE/'analysis/rounding_summary.csv',summary)
    limit=[]
    for sid in sorted({r['scenario_id'] for r in rows if r['algorithm']=='milp' and r['optimum'] is None}):
        ds=[d for d in details if d['scenario_id']==sid and d['algorithm']=='milp']
        u=min(len(d['selected_original_ids']) for d in ds)
        lower=max(d['info']['dual_bound'] for d in ds)
        rr=next(r for r in rows if r['scenario_id']==sid and r['algorithm']=='milp')
        limit.append({'scenario_id':sid,'density':rr['density'],'seed':rr['seed'],
                      'lp_bound':rr['lp_bound'],'best_dual_bound':lower,
                      'integer_lower_bound':int(np.ceil(lower-1e-6)),'best_incumbent':u})
    write_csv(HERE/'analysis/time_limited_instances.csv',limit)
    stability=[]
    for family in ('arithmetic','width_sensitivity','synthetic'):
        for method in METHODS:
            sub=[r for r in rows if r['family']==family and r['algorithm']==method]
            relative=[(r['solver_ms_max']-r['solver_ms_min'])/r['solver_ms_median'] for r in sub]
            stability.append({'family':family,'algorithm':method,
                              'median_relative_timing_range':float(np.median(relative)),
                              'fractional_lp_cases':sum(r['lp_fractional_variables']>0 for r in sub),
                              'median_lp_integrality_gap':stats(sub,'lp_integrality_gap')['median'],
                              'median_full_break_even':stats(sub,'break_even_replays_full')['median'],
                              'median_cached_break_even':stats(sub,'break_even_replays_cached_matrix')['median']})
    write_csv(HERE/'analysis/stability.csv',stability)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
                         'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    syn=[r for r in rows if r['family']=='synthetic']
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.25),layout='constrained')
    vals=[[r['optimality_gap_pct'] for r in syn if r['algorithm']==m and r['optimality_gap_pct'] is not None] for m in METHODS]
    box=axs[0].boxplot(vals,tick_labels=['Greedy','LP','IP'],patch_artist=True,widths=.5)
    for patch,m in zip(box['boxes'],METHODS):patch.set_facecolor(COLORS[m]);patch.set_alpha(.7)
    axs[0].set(ylabel='Excess suite size (%)',title='(a) Quality: 34 proved references')
    times=[stats([r for r in syn if r['algorithm']==m],'solver_ms_median')['median'] for m in METHODS]
    axs[1].bar(['Greedy','LP','IP'],times,color=[COLORS[m] for m in METHODS],width=.55)
    axs[1].set(yscale='log',ylabel='Median solve time (ms)',title='(b) Time: all 40 stress instances')
    for i,v in enumerate(times):axs[1].text(i,v*1.25,f'{v:.2f}',ha='center',fontsize=7)
    axs[1].set_ylim(.03,1500)
    for ax in axs:ax.grid(axis='y',alpha=.2)
    save(fig,'stress_tradeoff')
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.15),layout='constrained')
    main=[r for r in rows if r['family']=='arithmetic' and r['mutant_count']=='all']
    for method,marker,style in zip(METHODS,('o','s','x'),('-','--',':')):
        xs=[300,1200,4800]
        ys=[stats([r for r in main if r['algorithm']==method and r['n_original']==n],'size')['median'] for n in xs]
        axs[0].plot(range(3),ys,marker=marker,linestyle=style,label=NAMES[method],color=COLORS[method],ms=4)
    axs[0].set_xticks(range(3),['300','1,200','4,800']);axs[0].set(xlabel='Original candidate tests',ylabel='Median selected tests',title='(a) RV32 models; 65 mutants')
    axs[0].legend(loc='upper right',frameon=False)
    for method,marker,style in zip(METHODS,('o','s','x'),('-','--',':')):
        xs=[300,1200,4800]
        ys=[stats([r for r in main if r['algorithm']==method and r['n_original']==n],'selection_setup_ms')['median'] for n in xs]
        axs[1].plot(range(3),ys,marker=marker,linestyle=style,color=COLORS[method],ms=4)
    axs[1].set_xticks(range(3),['300','1,200','4,800']);axs[1].set(xlabel='Original candidate tests',ylabel='Compression + selection (ms)',title='(b) Shared preprocessing included')
    for ax in axs:ax.grid(alpha=.2)
    save(fig,'main_scaling')
    fig,axs=plt.subplots(1,2,figsize=(5.5,2.2),layout='constrained')
    audit=read_rows(HERE/'analysis/audit_unconstrained_variants.csv')
    order=(*METHODS,'random_pruned')
    values=[[r['excluded_retention_pct'] for r in audit if r['algorithm']==method] for method in order]
    box=axs[0].boxplot(values,tick_labels=['Greedy','LP','IP','Rand.+P'],
                       patch_artist=True,widths=.5,whis=(0,100),showfliers=False)
    for patch,color in zip(box['boxes'],[COLORS[x] for x in METHODS]+['#89959E']):
        patch.set_facecolor(color);patch.set_alpha(.7)
    axs[0].axhline(100,color='#555555',linestyle='--',linewidth=.8)
    axs[0].text(.98,.96,'Original suites: 100%',transform=axs[0].transAxes,ha='right',va='top',fontsize=6.5)
    axs[0].set(title='(a) Variants outside constraints',ylabel='Detection retained (%)',ylim=(0,108))
    for sid in ids[3:]:
        vals=[float(np.mean([r['size'] for r in ab if r['scenario_id']==sid and r['budget']==b])) for b in (1,4,16,64)]
        short=sid.replace('syn_','').replace('_s11','').replace('_n640','')
        axs[1].plot(range(4),vals,'o-',ms=3,label=short)
    axs[1].set_xticks(range(4),['1','4','16','64']);axs[1].set(title='(b) Rounding budget sensitivity',xlabel='Candidate budget',ylabel='Mean selected tests')
    axs[1].legend(fontsize=6.8,frameon=False,loc='upper right',bbox_to_anchor=(.98,.87))
    for ax in axs:ax.grid(alpha=.2)
    save(fig,'process_ablation')
    comparison_figure(rows,ab,ids)
    meta={'calls':len(ab),'fixed_matrices':len(ids),'rounding_seeds':list(range(101,111)),
          'budgets':[1,4,16,64],'coverage_checks':len(ab),'elapsed_seconds':time.perf_counter()-started,
          'note':'Budget 1 evaluates the deterministic near-one LP seed + repair. Seeds vary only randomized trials; these are not new data instances.'}
    if not reuse:
        (HERE/'analysis/supplement_manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'ablation':summary,'time_limited':limit,'stability':stability},indent=2))


def comparison_figure(rows,ab,ids):
    """Place all four diagnostic panels together without altering the recorded data."""
    syn=[r for r in rows if r['family']=='synthetic']
    audit=read_rows(HERE/'analysis/audit_unconstrained_variants.csv')
    fig,axes=plt.subplots(2,2,figsize=(5.5,3.4),layout='constrained')
    quality,timing,retention,budget=axes.flat
    gaps=[[r['optimality_gap_pct'] for r in syn if r['algorithm']==m and r['optimality_gap_pct'] is not None] for m in METHODS]
    box=quality.boxplot(gaps,tick_labels=['Greedy','LP','IP'],patch_artist=True,widths=.5,
                        whis=(0,100),showfliers=False,medianprops={'color':'#26394A'})
    for patch,m in zip(box['boxes'],METHODS):
        patch.set_facecolor(COLORS[m]);patch.set_alpha(.65)
    quality.set(title='(a) Stress quality: 34 references',ylabel='Excess suite size (%)')
    times=[stats([r for r in syn if r['algorithm']==m],'solver_ms_median')['median'] for m in METHODS]
    timing.bar(['Greedy','LP','IP'],times,color=[COLORS[m] for m in METHODS],width=.55)
    timing.set(title='(b) Stress time: all 40 cases',ylabel='Median solve time (ms)',yscale='log',ylim=(.03,1500))
    for i,v in enumerate(times):timing.text(i,v*1.25,f'{v:.2f}',ha='center',fontsize=7)
    order=(*METHODS,'random_pruned')
    values=[[r['excluded_retention_pct'] for r in audit if r['algorithm']==m] for m in order]
    box=retention.boxplot(values,tick_labels=['Greedy','LP','IP','Rand.+P'],patch_artist=True,
                          widths=.5,whis=(0,100),showfliers=False,medianprops={'color':'#26394A'})
    for patch,color in zip(box['boxes'],[COLORS[m] for m in METHODS]+['#89959E']):
        patch.set_facecolor(color);patch.set_alpha(.65)
    retention.axhline(100,color='#555555',linestyle='--',linewidth=.8)
    retention.text(.98,.8,'Original: 100%',transform=retention.transAxes,ha='right',fontsize=6.8)
    retention.set(title='(c) Excluded-variant retention',ylabel='Detection retained (%)',ylim=(0,110))
    for sid in ids[3:]:
        means=[float(np.mean([r['size'] for r in ab if r['scenario_id']==sid and r['budget']==b])) for b in (1,4,16,64)]
        label=sid.replace('syn_','').replace('_s11','').replace('_n640','')
        budget.plot(range(4),means,'o-',ms=3,label=label)
    budget.set_xticks(range(4),['1','4','16','64'])
    budget.set(title='(d) Rounding budget sensitivity',xlabel='Candidate budget',ylabel='Mean selected tests')
    budget.legend(fontsize=6.3,frameon=False,loc='upper right',bbox_to_anchor=(.99,.97))
    for ax in axes.flat:
        ax.grid(axis='y',alpha=.2);ax.set_title(ax.get_title(),fontsize=8.2)
    save(fig,'comparative_analysis')


def save(fig,name):
    for extension in ('pdf','png'):
        fig.savefig(HERE/'figures'/f'{name}.{extension}',dpi=220,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
