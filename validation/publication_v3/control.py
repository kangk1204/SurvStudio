"""Evidence gates: cost, fixed candidate selection, seal, and finite qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from study import ROOT, protocol, hashes, environment


def write(path,value):
    path=Path(path)
    if path.exists(): raise ValueError("Evidence cannot be overwritten")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def cost(summary):
    cfg=protocol()
    if summary.get("stage")!="cost" or not summary.get("complete"): raise ValueError("Complete planned cost study required")
    durations={}
    for row in summary["summaries"]:
        durations[(row["n"],row["p"])]=max(durations.get((row["n"],row["p"]),0),row["elapsed_p95_seconds"])
    # Deliberately conservative: the pilot runs both candidates, whereas confirmation runs one.
    seconds=sum(durations[(c["n"],c["p"])]*len(c["conditions"])*c["replicates"] for c in cfg["confirmation"])
    days=cfg["cost"]["safety_factor"]*seconds/cfg["cost"]["available_workers"]/86400
    return dict(status="feasible" if days<=cfg["cost"]["maximum_confirmation_days"] else "v2_fallback_resource_limit",
        conservative_days=days,workers_assumed=cfg["cost"]["available_workers"],safety_factor=cfg["cost"]["safety_factor"],
        paired_pilot_includes_both_candidates=True,limitation="Pilot has one clinical variable. Two-variable stress cost must be watched without changing its count or method.")


def select(summary):
    cfg=protocol()
    if summary.get("stage")!="selection" or not summary.get("complete"): raise ValueError("Complete independent selection study required")
    decisions=[]
    for candidate,label in zip(cfg["candidates"],("A","B")):
        rows=[r for r in summary["summaries"] if r["method"]==f"candidate_{label}_linear" and r["condition"] in cfg["supported"]["guarded_linear"]]
        if len(rows)!=12: raise ValueError("All twelve linear support cells required")
        eligible=all(r["missing"]==0 and r["failures"]==0 and r["fwer"]<=.06 and r["conditional_fwer"] is not None and r["conditional_fwer"]<=.06 for r in rows)
        healthy=[r for r in rows if r["condition"] in cfg["healthy_original"]]
        power=[r for r in rows if r["condition"].startswith("partial_")]
        availability=min(r["allowed_fraction"] for r in healthy)
        retention=min(r["power_ratio"]["point"] if r["power_ratio"] else 0 for r in power)
        eligible=eligible and availability>=.90 and retention>=.90
        seconds=float(np.mean([r["mean_diagnostic_seconds"] for r in rows]))
        decisions.append(dict(candidate=candidate,label=label,eligible=eligible,minimum_healthy_allowed=availability,
            minimum_power_ratio=retention,mean_diagnostic_seconds=seconds))
    available=[r for r in decisions if r["eligible"]]
    available.sort(key=lambda r:(-r["minimum_healthy_allowed"],-r["minimum_power_ratio"],r["mean_diagnostic_seconds"],r["label"]))
    return dict(status="selected" if available else "v2_fallback_no_eligible_candidate",selection_eligible=bool(available),
        candidate=None if not available else available[0]["candidate"],candidate_evaluations=decisions,
        confirmation_evidence=False,selection_seed=cfg["selection_seed"])


def seal(selection,reference):
    if not selection.get("selection_eligible") or reference.get("passed") is not True:
        raise ValueError("Selection and independent R must pass before seal")
    if reference.get("environment")!=environment(): raise ValueError("R reference runtime differs from frozen environment")
    for name,expected in reference["source_hashes"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected: raise ValueError("R-verified source changed")
    policy_path=ROOT/"src/survival_toolkit/data/marker_bootstrap_policy.json"
    policy=json.loads(policy_path.read_text())
    if policy["candidate"]!=selection["candidate"] or policy["selection_status"]!="selected_before_confirmation":
        raise ValueError("Commit selected global policy before seal")
    names=list(hashes())
    if subprocess.check_output(["git","-C",str(ROOT),"status","--porcelain","--",*names],text=True).strip():
        raise ValueError("Numerical sources must be committed before seal")
    return dict(candidate=selection["candidate"],selection_eligible=True,reference_passed=True,source_hashes=hashes(),
        environment=environment(),git_revision=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),
        qualification_status="pending_confirmation",selected_basis_policy="one candidate for both bases")


def qualify(summaries):
    cfg=protocol();required={"main","extension","large","stress"}
    if {s["stage"] for s in summaries}!=required or not all(s["complete"] for s in summaries): raise ValueError("All 149500 planned datasets required")
    if sum(s["planned_datasets"] for s in summaries)!=149500: raise ValueError("Wrong confirmation total")
    frozen={s["configuration"]["freeze_sha256"] for s in summaries}
    if len(frozen)!=1 or None in frozen: raise ValueError("Mixed or unsealed confirmation")
    rows=[r for summary in summaries for r in summary["summaries"]];profiles={}
    for method,basis,support in (("v3_linear","linear",cfg["supported"]["guarded_linear"]),("v3_spline","restricted_cubic_spline",cfg["supported"]["guarded_spline"])):
        problems=[]
        main=[r for r in rows if r["stage"]=="main" and r["method"]==method and r["condition"] in support]
        for row in main:
            key=f'{row["condition"]}:N{row["n"]}:P{row["p"]}'
            if row["fwer_upper95"]>.06 or row["conditional_upper95"] is None or row["conditional_upper95"]>.06: problems.append(key+":fwer")
            if not row["condition"].startswith("partial_") and row["allowed_lower95"]<.90: problems.append(key+":availability")
            if row["condition"].startswith("partial_") and (not row["power_ratio"] or row["power_ratio"]["lower95"]<.90): problems.append(key+":power")
        for row in rows:
            if row["method"]!=method: continue
            baseline=next(r for r in rows if all(r[k]==row[k] for k in ("stage","condition","n","p")) and r["method"]==method.replace("v3_","v2_"))
            if row["stage"]=="main" and row["condition"] in cfg["misspecification_sensitivity"] and row["withhold_fraction"]<baseline["withhold_fraction"]-.05:
                problems.append(f'{row["condition"]}:P{row["p"]}:sensitivity_loss')
            healthy=row["condition"] in (cfg["healthy_stress"] if row["stage"]=="stress" else cfg["healthy_original"])
            if row["stage"]!="main" and healthy and (row["fwer_lower95"]>.06 or (row.get("conditional_lower95") or 0)>.06):
                problems.append(f'{row["stage"]}:{row["condition"]}:P{row["p"]}:clear_extension_failure')
        profiles[basis]=dict(status="passed_supported_conditions_only" if not problems else "failed_exploratory_only",
            supported_conditions=support,failed_conditions=problems,qualification_dimensions=[{"n":180,"p":30},{"n":180,"p":300}])
    return dict(method_version="marker-inference/3",study_complete=True,profiles=profiles,freeze_sha256=next(iter(frozen)),
        default_policy_eligible=profiles["linear"]["status"]=="passed_supported_conditions_only",
        production_promotion="Requires final source-bound CI and package review; this file alone does not promote")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["cost","select","seal","qualify"])
    parser.add_argument("inputs",nargs="+");parser.add_argument("--output",required=True)
    args=parser.parse_args();values=[json.loads(Path(p).read_text()) for p in args.inputs]
    result=cost(values[0]) if args.command=="cost" else select(values[0]) if args.command=="select" else seal(*values) if args.command=="seal" else qualify(values)
    result["input_hashes"]={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.inputs}
    write(args.output,result);print(json.dumps({k:v for k,v in result.items() if k in {"status","conservative_days","candidate","default_policy_eligible"}}))
