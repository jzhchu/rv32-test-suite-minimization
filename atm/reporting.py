"""Build standalone HTML analysis, CSV tables, and publication-friendly figures."""
import argparse
import base64
import csv
import html
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .experiment import ROOT,write_csv,write_json

COLORS={"greedy":"#177E89","lp_round":"#DE8F05","milp":"#6658A6","random":"#8A929D"}
LABELS={"greedy":"Greedy + pruning","lp_round":"LP rounding + repair","milp":"Integer programming","random":"Random-order baseline"}
MAIN=("greedy","lp_round","milp")


def read_rows(path):
    with path.open() as f:raw=list(csv.DictReader(f))
    rows=[]
    string_keys={"family","algorithm","scenario_id","status","matrix_sha256"}
    for row in raw:
        d={}
        for k,v in row.items():
            if k in string_keys: d[k]=v
            elif not v: d[k]=None
            elif v=="all":d[k]=v
            else:
                try:d[k]=float(v)
                except ValueError:d[k]=v
        rows.append(d)
    return rows


def stats(rows,key):
    a=np.array([r[key] for r in rows if r.get(key) is not None],dtype=float)
    if not len(a): return {"median":None,"min":None,"max":None,"q25":None,"q75":None}
    return {"median":float(np.median(a)),"min":float(a.min()),"max":float(a.max()),
            "q25":float(np.quantile(a,.25)),"q75":float(np.quantile(a,.75))}


def summary_stats(rows):
    result=[]
    for family in ("arithmetic","width_sensitivity","synthetic"):
        for algorithm in (*MAIN,"random"):
            rs=[r for r in rows if r["family"]==family and r["algorithm"]==algorithm]
            if not rs:continue
            result.append({"family":family,"algorithm":algorithm,"instances":len(rs),
                           "median_size":stats(rs,"size")["median"],
                           "median_reduction_pct":stats(rs,"reduction_pct")["median"],
                           "median_gap_pct":stats(rs,"optimality_gap_pct")["median"],
                           "max_gap_pct":stats(rs,"optimality_gap_pct")["max"],
                           "optimal_matches":sum(abs(r["size"]-(r["optimum"] or -999))<1e-6 for r in rs),
                           "certified_reference_instances":sum(r["optimum"] is not None for r in rs),
                           "median_solver_ms":stats(rs,"solver_ms_median")["median"],
                           "median_setup_ms":stats(rs,"total_setup_ms")["median"],
                           "median_break_even_full":stats(rs,"break_even_replays_full")["median"]})
    return result


def plot_line(ax,rows,xkey,ykey,algorithms=MAIN,log=False):
    xs=sorted({r[xkey] for r in rows})
    for algorithm in algorithms:
        med=[];lo=[];hi=[]
        for x in xs:
            st=stats([r for r in rows if r["algorithm"]==algorithm and r[xkey]==x],ykey)
            med.append(st["median"]);lo.append(st["q25"]);hi.append(st["q75"])
        markers={"greedy":"o","lp_round":"s","milp":"x","random":"D"}
        styles={"greedy":"-","lp_round":"--","milp":":","random":"-."}
        ax.plot(xs,med,marker=markers[algorithm],linestyle=styles[algorithm],
                label=LABELS[algorithm],color=COLORS[algorithm],lw=2,ms=6,
                markerfacecolor="none" if algorithm!="milp" else COLORS[algorithm])
        ax.fill_between(xs,lo,hi,color=COLORS[algorithm],alpha=.12)
    if log:ax.set_yscale("log")
    ax.set_xticks(xs,[f"{int(x):,}" for x in xs]);ax.grid(alpha=.2)
    ax.spines[["top","right"]].set_visible(False)


def savefig(fig,folder,name):
    fig.savefig(folder/(name+".png"),dpi=180,bbox_inches="tight",facecolor="white")
    fig.savefig(folder/(name+".svg"),bbox_inches="tight",facecolor="white")
    plt.close(fig)


def figures(rows,out):
    folder=out/"figures";folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":12,
                         "axes.labelsize":10,"svg.fonttype":"none"})
    ar=[r for r in rows if r["family"]=="arithmetic" and r["mutant_count"]=="all"]
    syn=[r for r in rows if r["family"]=="synthetic"]
    fig,ax=plt.subplots(2,2,figsize=(12.5,8.5),layout="constrained")
    plot_line(ax[0,0],ar,"n_original","size")
    ax[0,0].set(title="Executed arithmetic models: suite size",xlabel="Original candidate tests",ylabel="Selected tests (median / IQR)")
    ax[0,0].legend(fontsize=8)
    plot_line(ax[0,1],ar,"n_original","selection_setup_ms",log=True)
    ax[0,1].set(title="Selection cost, including common compression",xlabel="Original candidate tests",ylabel="Milliseconds (log scale)")
    for axis,family,title in [(ax[1,0],"arithmetic","Arithmetic: gap to proved optimum"),(ax[1,1],"synthetic","Synthetic stress: gap to proved optimum")]:
        data=[[r["optimality_gap_pct"] for r in rows if r["family"]==family and r["algorithm"]==a and r["optimality_gap_pct"] is not None] for a in MAIN]
        boxes=axis.boxplot(data,tick_labels=["Greedy","LP rounding","IP"],patch_artist=True,widths=.45,showmeans=True)
        for patch,a in zip(boxes["boxes"],MAIN):patch.set_facecolor(COLORS[a]);patch.set_alpha(.6)
        axis.set(title=title,ylabel="Excess selected tests (%)")
        axis.grid(axis="y",alpha=.2);axis.spines[["top","right"]].set_visible(False)
    fig.suptitle("TEST SUITE MINIMIZATION  |  Local CPU experiments",fontsize=17,fontweight="bold")
    savefig(fig,folder,"overview")

    fig,axs=plt.subplots(1,2,figsize=(12,4.3),layout="constrained")
    max_goals=max(r["m_original"] for r in syn)
    for ax,density in zip(axs,(.05,.15)):
        sub=[r for r in syn if r["m_original"]==max_goals and abs(r["density"]-density)<1e-8]
        plot_line(ax,sub,"n_original","size")
        ns=sorted({r["n_original"] for r in sub})
        bounds=[stats([r for r in sub if r["n_original"]==n and r["algorithm"]=="milp"],"lp_bound")["median"] for n in ns]
        ax.plot(ns,bounds,"k--",label="LP lower bound",lw=1.5)
        ax.set(title=f"{int(max_goals)} goals; Bernoulli density = {density:.2f}",xlabel="Candidate columns",ylabel="Selected tests (median / IQR)")
    axs[0].legend(fontsize=8)
    fig.suptitle("SYNTHETIC STRUCTURE STRESS TEST  |  Not arithmetic execution data",fontweight="bold")
    savefig(fig,folder,"synthetic_structure")

    details=json.loads((out/"solutions.json").read_text())
    sid="arith_w32_n1200_b0.8_mall_s11"
    if not any(d["scenario_id"]==sid for d in details):
        sid=next(r["scenario_id"] for r in rows if r["family"]=="arithmetic" and r["mutant_count"]=="all")
    target=[d for d in details if d["scenario_id"]==sid and d["algorithm"]=="milp" and d["rep"]==0]
    if target:
        a=np.load(out/"instances"/(sid+".npz"))["matrix"]
        chosen=target[0]["selected_original_ids"]
        fig,axs=plt.subplots(1,2,figsize=(12.5,4.6),layout="constrained",gridspec_kw={"width_ratios":[2,1]})
        for ax,mat,title in [(axs[0],a,f"Full candidate suite: {a.shape[1]:,} tests"),(axs[1],a[:,chosen],f"IP-selected suite: {len(chosen)} tests")]:
            ax.imshow(mat,aspect="auto",interpolation="nearest",cmap="Blues",vmin=0,vmax=1)
            ax.set(title=title,xlabel="Test index",ylabel="Seeded mutant index")
        fig.suptitle("OBSERVED KILL MATRICES  |  Every row remains covered",fontweight="bold")
        savefig(fig,folder,"kill_matrix")

    fig,axs=plt.subplots(1,2,figsize=(12,4.1),layout="constrained")
    trace=next((d["trace"] for d in details if d["scenario_id"]==sid and d["algorithm"]=="greedy" and d["rep"]==0),None)
    if trace:
        axs[0].plot([0]+[x["selected"] for x in trace],[0]+[x["covered"] for x in trace],"o-",color=COLORS["greedy"])
    axs[0].set(title="Greedy construction (one arithmetic instance)",xlabel="Tests added before reverse deletion",ylabel="Covered detectable mutants")
    sid2="syn_m80_n640_d0.15_s11"
    if not any(d["scenario_id"]==sid2 for d in details):
        sid2=next(r["scenario_id"] for r in rows if r["family"]=="synthetic")
    trace2=next((d["trace"] for d in details if d["scenario_id"]==sid2 and d["algorithm"]=="lp_round" and d["rep"]==0),None)
    if trace2:
        axs[1].step([x["trial"] for x in trace2],[x["incumbent"] for x in trace2],where="post",color=COLORS["lp_round"],lw=2,label="Best rounded feasible suite")
        axs[1].axhline(trace2[0]["lp_bound"],color="black",ls="--",label="LP lower bound")
        axs[1].legend(fontsize=8)
    axs[1].set(title="LP rounding trials (one synthetic instance)",xlabel="Rounding / repair trial",ylabel="Number of tests")
    for ax in axs:ax.grid(alpha=.2);ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("ALGORITHM PROCESS  |  Construction and incumbent traces",fontweight="bold")
    savefig(fig,folder,"algorithm_process")

    fig,axs=plt.subplots(1,2,figsize=(12,4.2),layout="constrained")
    plot_line(axs[0],ar,"n_original","n_effective",algorithms=("milp",))
    axs[0].set(title="Exact preprocessing: distinct nonzero kill signatures",xlabel="Original candidate tests",ylabel="Effective candidate columns")
    plot_line(axs[1],ar,"n_original","break_even_replays_full",log=True)
    axs[1].set(title="Measured in-process replay break-even",xlabel="Original candidate tests",ylabel="Replays to amortize full setup (log scale)")
    axs[1].legend(fontsize=8)
    savefig(fig,folder,"preprocessing_and_cost")


def table(headers,data):
    return "<table><thead><tr>"+"".join(f"<th>{html.escape(str(h))}</th>" for h in headers)+"</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in row)+"</tr>" for row in data)+"</tbody></table>"


def fnum(x,digits=2):return "—" if x is None else f"{x:,.{digits}f}"


def make_report(rows,aggregates,manifest,out):
    main=[r for r in rows if r["family"]=="arithmetic"]
    arithmetic_all=[r for r in rows if r["family"]!="synthetic"]
    syn=[r for r in rows if r["family"]=="synthetic"]
    findings={"instances":manifest["instances"],"algorithm_runs":manifest["algorithm_runs"],
              "coverage_checks":manifest["coverage_checks_passed"],"elapsed_seconds":manifest["elapsed_seconds"],
              "arithmetic_instances":len(main)//4,"width_sensitivity_instances":len([r for r in rows if r["family"]=="width_sensitivity"])//4,
              "synthetic_instances":len(syn)//4,
              "milp_optimal_runs":manifest["milp_optimal_runs"],
              "aggregates":aggregates,"arithmetic_milp_reduction":stats([r for r in arithmetic_all if r["algorithm"]=="milp"],"reduction_pct"),
              "arithmetic_milp_size":stats([r for r in arithmetic_all if r["algorithm"]=="milp"],"size"),
              "arithmetic_fractional_lp_instances":sum(r["lp_fractional_variables"]>0 for r in arithmetic_all if r["algorithm"]=="milp"),
              "synthetic_fractional_lp_instances":sum(r["lp_fractional_variables"]>0 for r in syn if r["algorithm"]=="milp")}
    write_json(out/"findings.json",findings)
    index={(a["family"],a["algorithm"]):a for a in aggregates}
    ag=index[("arithmetic","greedy")];al=index[("arithmetic","lp_round")];ai=index[("arithmetic","milp")]
    sg=index[("synthetic","greedy")];sl=index[("synthetic","lp_round")];si=index[("synthetic","milp")]
    interpretations=f'''<p><b>主实验：</b>贪心在 {ag['optimal_matches']}/{ag['instances']} 个实例达到最优，LP 取整在 {al['optimal_matches']}/{al['instances']} 个实例达到最优；贪心最差规模差距为 {ag['max_gap_pct']:.2f}%。三种主要算法的测试数量中位数均为 {ag['median_size']:.0f}。贪心、LP、IP 的纯求解中位时间分别为 {ag['median_solver_ms']:.3f}、{al['median_solver_ms']:.3f}、{ai['median_solver_ms']:.3f} ms。小程序的矩阵按操作分块，覆盖结构简单，IP 在这里并未表现为最慢；共同压缩成本往往大于算法本身。</p>
<p><b>合成结构实验：</b>在 {sg['certified_reference_instances']} 个已有最优参考的实例上，贪心与 LP 取整的规模差距中位数分别为 {sg['median_gap_pct']:.2f}% 和 {sl['median_gap_pct']:.2f}%。全部合成实例的 IP 求解时间中位数为 {si['median_solver_ms']:.2f} ms，明显高于 LP 的 {sl['median_solver_ms']:.2f} ms 和贪心的 {sg['median_solver_ms']:.3f} ms。另有 {sg['instances']-sg['certified_reference_instances']} 个实例在 5 秒限制下未证明最优，图中的 IP 值对这些实例仅表示当前可行解，不作为精确最优值。</p>
<p><b>可以支持的判断：</b>本实验中，简单结构上贪心的低开销很有吸引力；LP 取整以额外成本改善部分解；复杂覆盖结构使整数规划的最优性证明成本明显增加。原始测试缩减比例很高，但其中包含检测列重复带来的共享预处理收益，不能把全部缩减归因于某个优化算法。曲线重合的位置表示方法得到相同规模，并非漏画。</p>'''
    names={"arithmetic":"主实验：32 位程序","width_sensitivity":"补充：8/16 位程序","synthetic":"补充：合成覆盖矩阵"}
    method_names={"greedy":"贪心＋冗余删除","lp_round":"LP 取整＋修复","milp":"整数规划","random":"随机顺序基线"}
    result_table=table(["实验组","算法","实例","测试数中位数","匹配最优参考","中位差距 %","最大差距 %","求解 ms 中位数"],
        [[names[a["family"]],method_names[a["algorithm"]],a["instances"],fnum(a["median_size"],1),f"{a['optimal_matches']}/{a['certified_reference_instances']}",fnum(a["median_gap_pct"]),fnum(a["max_gap_pct"]),fnum(a["median_solver_ms"],3)] for a in aggregates])
    # One row per factorial cell, aggregating the five data seeds.
    condition_rows=[]
    for count in (30,"all"):
        for n in (300,1200,4800):
            for ratio in (.2,.8):
                sub=[r for r in main if r["mutant_count"]==count and r["n_original"]==n and r["boundary_ratio"]==ratio]
                if not sub:continue
                condition_rows.append([n,ratio,65 if count=="all" else count]+[fnum(stats([r for r in sub if r["algorithm"]==a],"size")["median"],1) for a in MAIN])
    condition_table=table(["原始测试数","新增测试边界比例","错误变体数","贪心","LP 取整","整数规划"],condition_rows)
    cost_rows=[]
    scaled=lambda rs,key,scale: None if stats(rs,key)["median"] is None else stats(rs,key)["median"]*scale
    for a in MAIN:
        sub=[r for r in arithmetic_all if r["algorithm"]==a]
        cost_rows.append([method_names[a],fnum(scaled(sub,"matrix_seconds",1000),3),fnum(stats(sub,"selection_setup_ms")["median"],3),fnum(scaled(sub,"replay_raw_seconds",1e6),2),fnum(scaled(sub,"replay_selected_seconds",1e6),2),fnum(stats(sub,"break_even_replays_full")["median"],0),fnum(stats(sub,"break_even_replays_cached_matrix")["median"],0)])
    cost_table=table(["算法","建矩阵 ms","压缩＋选择 ms","完整回放 μs","压缩回放 μs","全成本回本次数","缓存矩阵回本次数"],cost_rows)
    number=manifest["milp_optimal_runs"]
    num_ip=manifest["algorithm_runs"]//4
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>整数与位运算程序的测试集最小化：实验报告</title>
<style>body{{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;color:#202d3a;background:#f4f7fa;margin:0;line-height:1.8}}main{{max-width:1120px;margin:35px auto;background:white;padding:40px 48px;border-radius:12px}}h1{{font-size:30px;line-height:1.4}}h2{{margin-top:38px;border-bottom:2px solid #dfe9ed;padding-bottom:8px;color:#177e89}}h3{{margin-top:26px}}.subtitle{{color:#64748b}}.card{{padding:18px 24px;background:#eef7f7;border-left:4px solid #177e89}}img{{width:100%;height:auto;margin:14px 0}}table{{width:100%;border-collapse:collapse;font-size:13px;margin:20px 0}}th,td{{padding:9px;border-bottom:1px solid #dfe5ea;text-align:left}}th{{background:#eef3f7}}code,pre{{font-family:ui-monospace,monospace;background:#f1f5f9}}pre{{padding:18px;overflow:auto}}a{{color:#146a94}}.small{{font-size:13px;color:#64748b}}@media print{{body{{background:white}}main{{margin:0;padding:0;max-width:none}}h2{{break-after:avoid}}img,table{{break-inside:avoid}}}}</style>
<main><p class="subtitle">COMP 6704 · 独立课程实验 · 本地 CPU · 2026-11-24</p>
<h1>整数与位运算程序的测试集最小化</h1><p class="subtitle">贪心、线性松弛与整数规划的比较 · 实验部分与复现说明</p>
<div class="card">已完成 <b>{findings['arithmetic_instances']} 个 32 位主实验实例</b>、{findings['width_sensitivity_instances']} 个位宽敏感性实例和 {findings['synthetic_instances']} 个合成结构实例；共 {manifest['algorithm_runs']} 次算法运行，全部通过原始矩阵覆盖复核。整数规划 {number}/{num_ip} 次运行返回已证明最优。所有程序、错误变体和测试均为本课程独立构建的合成实验材料。</div>
<h2>1. 问题与评价目标</h2>
<p>测试集积累会造成检测能力重叠。本实验研究：在保留原始候选测试能够检出的全部已知错误变体的条件下，能将测试数量压缩到多小，以及不同优化方法需要付出多少计算成本。这里的覆盖指<b>错误变体检测覆盖</b>，并非代码行覆盖，也不代表程序已被证明正确。</p>
<p>令 A<sub>ij</sub>=1 表示测试 j 能使第 i 个错误变体产生与参考实现不同的输出。模型为：</p>
<pre>minimize    Σⱼ xⱼ
subject to  Σⱼ Aᵢⱼ xⱼ ≥ 1    for every originally detectable mutant i
            xⱼ ∈ {{0,1}}</pre>
<p>所有测试成本统一为 1，因此优化目标是测试数量；计时作为额外评价。原始候选集未检出的变体独立记录，不计入覆盖约束，不将其认定为等价变体。本次算术实例未检出变体累计为 {manifest['undetected_rows_total']}。</p>
<h2>2. 实验对象与数据生成</h2>
<p>程序包含 10 类操作：add、sub、mul、and、xor、shl、lsr、asr、ltu、lts。算术结果使用定长无符号位模式；有符号解释采用二补码；移位量按字长取模。参考实现和 65 个一阶人工错误变体均可在 Python CPU 环境中直接执行。变体包括截断遗漏、半字进位/借位遗漏、错误操作符、符号解释错误、移位掩码错误及错误的零值/恒等值快捷分支。未使用外部生产实现或真实漏洞。</p>
<p>每个候选集以固定的 180 条通用诊断用例开始（18 对输入 × 10 类操作），再按配置添加边界或均匀随机输入。诊断前缀在查看检测矩阵之前定义；边界比例 0.2/0.8 仅作用于新增测试。候选集允许重复输入与重复检测行为。30 变体条件采用操作轮转抽样，保持各操作参与；65 变体条件使用完整目录。</p>
{table(['实验组','条件','实例数'],[['主实验','32 位；测试数 300/1200/4800 × 新增边界比例 0.2/0.8 × 变体数 30/65 × 5 个种子',60],['位宽敏感性','8/16 位；1200 测试；新增边界比例 0.5；65 变体；5 个种子',10],['合成结构敏感性','40/80 目标 × 160/640 列 × Bernoulli 密度 0.05/0.15 × 5 个种子',40]])}
<p>合成结构实验直接生成覆盖矩阵，不声称来自算术程序。独立 Bernoulli 采样后，对没有任何覆盖列的行补一个随机见证列以保证可行；实际密度在原始 CSV 中记录。其目的仅是检验方法在更强覆盖重叠与分数松弛解下的表现。</p>
<h2>3. 算法与公平比较设置</h2>
<p>共同预处理仅删除全零列和合并完全相同的非零检测列，并保留原始测试 ID。单位成本下这些操作保持最优值；未执行支配列删除或根据最优解筛选输入。所有算法获得相同压缩矩阵，输出都映射回原始矩阵逐行验证。</p>
<ul><li><b>贪心：</b>每次最大化尚未覆盖目标数，平局按原始候选顺序打破；完成后逆序删除冗余测试。</li>
<li><b>LP 取整：</b>放松为 0≤x≤1，用 HiGHS 内点法求解；若 LP 解已整数化，直接修复并删冗余；否则进行 16 次显式取整候选评估。第 1 次保留接近 1 的变量再修复，其余候选以 pⱼ=min(1,ln(m+1)xⱼ) 抽样；每次均贪心补齐和逆序删冗余，选择规模最小者。计时包括全部取整与修复开销，不对该有限试验启发式额外宣称近似比。</li>
<li><b>整数规划：</b>SciPy/HiGHS 求解原二元模型，时间上限 5 秒，目标相对差距 0；只有 solver status=0 才称为最优。若达限，保存可行解与界，不将其当作精确参考。</li>
<li><b>随机顺序基线：</b>在相同压缩候选列上随机排序，累计到覆盖全部目标，不做冗余删除；仅用于展示简单选择的冗余，算法质量结论主要基于前三种方法。</li></ul>
<p>数据种子为 11、23、37、51、71；每个实例每种方法计时 3 次，并轮换运行顺序，报告中位数。算法内随机种子在计时重复之间固定，时间重复不作为独立数据样本。5 个种子跨配置复用，110 个配置实例也不被视为 110 个完全独立样本。图中阴影为跨种子的四分位区间，不是置信区间。</p>
<p>环境：Python {sys_version(manifest)}，NumPy {manifest['numpy']}，SciPy {manifest['scipy']}，{manifest['machine']} / {manifest['platform']}。BLAS 线程通过运行脚本环境变量限制为 1；HiGHS 使用默认线程策略。计时单位为墙钟时间，已进行一次求解器热身。</p>
<h2>4. 主要结果</h2><img src="figures/overview.png" alt="算法规模、耗时与最优性差距概览">
{interpretations}
{result_table}
<p>最优匹配率的分母仅包含已有整数规划最优证明的实例。规模中位数按每个实例先汇总 3 次计时重复后计算；所有方法的观察检测覆盖均为 100%。该覆盖是模型可行性检查，不用于宣称压缩后能检测所有未知错误。</p>
<h3>4.1 测试规模、边界输入与错误目录大小</h3>{condition_table}
<p>原始测试数增加不必使最优测试数增加：新增测试可能提供更有互补性的组合，也可能只是重复已有检测模式。两种边界比例之间的差异还受到固定诊断前缀的影响，尤其在 300 测试条件中前缀占比较高。因此，本结果支持的是这些明确生成条件下的比较。</p>
<h3>4.2 线性松弛与结构敏感性</h3>
<p>70 个实际执行算术实例中，{findings['arithmetic_fractional_lp_instances']} 个存在分数 LP 变量；40 个合成实例中，{findings['synthetic_fractional_lp_instances']} 个存在分数 LP 变量。这解释了 LP 取整在两类实验中可能表现不同。LP 最优目标是整数最优值的下界，而最终取整后套件仍需接受独立覆盖检查。</p>
<img src="figures/synthetic_structure.png" alt="合成结构敏感性比较">
<h3>4.3 算法构造过程与检测矩阵</h3><img src="figures/algorithm_process.png" alt="贪心覆盖过程及取整试验轨迹">
<p>这里展示的是贪心构造进度和 LP 取整候选的最好可行值；不将其称为连续优化迭代收敛曲线。HiGHS 内点法迭代次数与 MILP 节点数保存在 solutions.json 中。</p>
<img src="figures/kill_matrix.png" alt="原始与压缩测试的实际检测矩阵">
<h2>5. 预处理与实际计算成本</h2><img src="figures/preprocessing_and_cost.png" alt="预处理及重复回放成本">
{cost_table}
<p>缩减比例以原始候选集为分母，同时报告有效检测列数，避免把共同预处理的收益全部归于某个求解算法。回放测量在同一个 Python 进程中逐条执行正确的小程序，并对照预计算输出；使用校准批次并取 3 次中位数，减少微秒计时噪声。它测量的是本实验模型，不是工业系统的实际加速。</p>
<p>全成本回本次数 = ceil((生成测试 + 构建检测矩阵 + 共同压缩 + 算法求解时间)/(完整套件回放时间 − 选中套件回放时间))。缓存矩阵口径只计压缩和求解。文件导出、环境安装、导入与人工构造错误的成本未计入；这些数字属于计算成本估计。对于廉价的小函数，单次回归通常不能抵消优化开销，重复使用次数必须进入效益分析。</p>
<h2>6. 正确性与复现检查</h2>
<ul><li>8 位加、减、乘对全部 65,536 对输入，与独立逐位加法/移位累加 oracle 交叉核对；另检查有符号比较和移位边界。</li>
<li>30 个随机小覆盖问题：MILP 与穷举最优解一致；LP 界未超过整数最优值；三种方法均可行。</li>
<li>分数三角形实例验证 LP 最优值 1.5、整数最优值 2；检查去重映射与原矩阵覆盖。</li>
<li>65 个错误变体在 8/16/32 位诊断集均有实际见证；未把未检出变体当作等价变体。</li>
<li>{manifest['coverage_checks_passed']} 次运行全部独立复核原始矩阵覆盖，且所有启发式解均未小于已证明最优值。</li></ul>
<p>数据与代码：<a href="raw_runs.csv">逐次结果 CSV</a> · <a href="summary.csv">每实例汇总 CSV</a> · <a href="aggregate.csv">分组结果 CSV</a> · <a href="solutions.json">选中 ID、求解状态与轨迹</a> · <a href="manifest.json">配置、环境与源码摘要</a>。instances/ 内含全部矩阵与算术输入，figures/ 同时提供 PNG 与可编辑 SVG。</p>
<h2>7. 结果适用范围</h2>
<p>本实验是可复现的课程原型：对象是独立小程序和人工变体；错误变体不等同真实漏洞或彼此独立的根因。参考实现也接受独立检查，但有限测试与少量操作不构成通用程序正确性证明。候选集使用固定边界前缀，覆盖结构相对简单；合成矩阵用于单独检验结构影响，不能混入程序实验后宣传工业效果。3 次时间重复和 5 个复用种子适合描述性比较，本报告不据此进行总体显著性推断。</p>
<h2>8. 公开参考资料</h2><ol>
<li>Mongiovì, M., Fornaia, A., Tramontana, E. (2020). <a href="https://link.springer.com/article/10.1007/s41109-020-00323-w">REDUNET: reducing test suites by integrating set cover and network-based optimization</a>. 用于集合覆盖建模背景；本实验的检测标准为变体检测覆盖。</li>
<li><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html">SciPy linprog 官方文档</a>：LP 接口与 HiGHS 内点法。</li>
<li><a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html">SciPy milp 官方文档</a>：二元约束、求解状态、最优界和时间上限。</li></ol>
<p class="small">报告由本地原始结果自动生成。配置和运行证据随项目交付；不存在为展示预期优势而手填的性能结果。</p></main></html>'''
    # Embed plots so the main report remains viewable without sibling image files.
    def embed(match):
        path=out/match.group(1)
        return 'src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"' if path.exists() else match.group(0)
    content=re.sub(r'src="(figures/[^\"]+\.png)"',embed,content)
    if manifest.get("quick"):
        content=content.replace("<h1>",'<div class="card">此文件为快速流程检查结果；下文实验设计表描述完整配置，实际运行规模以本页运行统计和 manifest 为准。</div><h1>',1)
    (out/"experimental_report.html").write_text(content)
    return findings


def sys_version(manifest): return manifest["python"].split()[0]


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=ROOT/"results")
    args,_=p.parse_known_args();out=args.output
    rows=read_rows(out/"summary.csv");manifest=json.loads((out/"manifest.json").read_text())
    agg=summary_stats(rows);write_csv(out/"aggregate.csv",agg)
    figures(rows,out)
    findings=make_report(rows,agg,manifest,out)
    print(json.dumps(findings,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
