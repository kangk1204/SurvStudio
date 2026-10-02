"""Audit raw replicate counts and render the expanded conditional-null stress test."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

LABELS = {
    "independent_null": "Independent markers", "linear_null": "Linear relation to Z",
    "heteroscedastic_null": "Heteroscedastic relation", "nonlinear_null": "Nonlinear relation to Z",
    "correlated_null": "Correlated markers", "covariate_dependent_censoring": "Censoring depends on Z",
    "partial_null_weak": "5 true markers: beta 0.2", "partial_null_strong": "5 true markers: beta 0.4",
}
METHODS = {"smith_linear": ("Residual permutation", "#0072B2", "o"),
           "raw_linear": ("Raw permutation", "#D55E00", "s"),
           "smith_quadratic": ("Residual + clinical Z²", "#009E73", "^")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    table=pd.read_csv(args.input/"replicates.csv")
    saved=json.loads((args.input/"summary.json").read_text())
    config=saved["settings"]["config"]
    assert not table.duplicated(["scenario","replicate","method"]).any()
    assert table.error.isna().all(), "Failures must be displayed rather than silently excluded"
    assert set(table.scenario)==set(LABELS) and set(table.method)==set(METHODS)
    source=[]
    for scenario in LABELS:
        for method in METHODS:
            group=table[(table.scenario==scenario)&(table.method==method)]
            assert len(group)==config["replicates"]
            flags=group.step_down_any_false
            assert flags.dtype==bool
            count=int(flags.sum());rate=count/len(group)
            low,high=proportion_confint(count,len(group),method="wilson")
            expected=next(r for r in saved["summary"] if r["scenario"]==scenario and r["method"]==method)["step_down"]
            assert expected["any_false_count"]==count and np.allclose(expected["fwer_mc95"],[low,high],atol=1e-14,rtol=0)
            power=group.step_down_power.dropna()
            mean=float(power.mean()) if len(power) else None
            se=float(power.std(ddof=1)/np.sqrt(len(power))) if len(power) else None
            source.append({"scenario":scenario,"method":method,"datasets":len(group),"false_rejection_datasets":count,
                           "fwer":rate,"mc_lower":low,"mc_upper":high,"mean_fdp":group.step_down_fdp.mean(),
                           "mean_power":mean,"power_mc_lower":None if mean is None else max(0,mean-1.96*se),
                           "power_mc_upper":None if mean is None else min(1,mean+1.96*se)})
    source=pd.DataFrame(source);args.output.mkdir(parents=True,exist_ok=True)
    source.to_csv(args.output/"source_data.csv",index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,"svg.fonttype":"none",
                         "svg.hashsalt":"survstudio-expanded-20261002"})
    fig,axes=plt.subplots(1,2,figsize=(6.7,4.5),gridspec_kw={"width_ratios":[1.4,1]})
    for number,(method,(label,color,marker)) in enumerate(METHODS.items()):
        offset=(number-1)*.18
        rows=source[source.method==method].set_index("scenario").loc[list(LABELS)]
        axes[0].errorbar(rows.fwer*100,np.arange(8)[::-1]+offset,
                         xerr=np.vstack([(rows.fwer-rows.mc_lower)*100,(rows.mc_upper-rows.fwer)*100]),
                         fmt=marker,color=color,markersize=4,capsize=2,lw=.9,label=label)
        rows=rows.loc[["partial_null_weak","partial_null_strong"]]
        axes[1].errorbar(rows.mean_power*100,np.array([1,0])+offset,
                         xerr=np.vstack([(rows.mean_power-rows.power_mc_lower)*100,(rows.power_mc_upper-rows.mean_power)*100]),
                         fmt=marker,color=color,markersize=4,capsize=2,lw=.9)
    axes[0].axvline(5,color="#777777",ls="--",lw=.9)
    axes[0].set(yticks=np.arange(8)[::-1],yticklabels=list(LABELS.values()),xlim=(0,10),ylim=(-.6,7.6),xlabel="Any false rejection (%)")
    axes[1].set(yticks=[1,0],yticklabels=["Weak signal","Strong signal"],xlim=(0,65),ylim=(-.5,1.5),xlabel="True markers detected (%)")
    axes[0].set_title("a  Family-wise rejection",loc="left",fontsize=9)
    axes[1].set_title("b  Partial-null power",loc="left",fontsize=9)
    for ax in axes:
        ax.spines[["top","right","left"]].set_visible(False);ax.tick_params(axis="y",length=0)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.54,.99),ncol=1,frameon=False,fontsize=8)
    fig.text(.02,.05,f"{config['replicates']:,} datasets per condition; {config['patients']} patients, {config['markers']} markers, {config['permutations']} permutations.",fontsize=7)
    fig.text(.02,.02,"Bars: pointwise 95% Monte Carlo intervals. This stress test does not establish universal error control.",fontsize=7)
    fig.subplots_adjust(left=.29,right=.98,bottom=.16,top=.76,wspace=.68)
    for suffix in ("png","pdf","svg"):
        meta={"Software":"SurvStudio audit"} if suffix=="png" else {"CreationDate":None,"ModDate":None} if suffix=="pdf" else {"Date":None}
        fig.savefig(args.output/f"expanded_calibration.{suffix}",dpi=300,metadata=meta)
    path=args.output/"expanded_calibration.svg";path.write_text(re.sub(r"<!DOCTYPE[^>]*>\s*","",path.read_text(),count=1))
    plt.close(fig)
    (args.output/"scientific_check.json").write_text(json.dumps({"duplicate_records":0,"failed_records":0,"records":len(table),
          "group_counts_verified":True,"wilson_intervals_independently_verified":True,
          "power_intervals":"Monte Carlo normal intervals from independent dataset-level mean power; markers within a dataset are not treated as independent"},indent=2)+"\n")


if __name__=="__main__":main()
