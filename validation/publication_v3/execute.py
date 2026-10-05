"""Persistent fixed-owner CPU execution, with cost/selection/source gates.

No dataset replacement, adaptive sample size, production promotion, submission,
web-tool automation, or human-declaration fabrication is performed here.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
PYTHON="/home/keunsoo/projects/SurvStudio-confirmation-v2-1ed6c10/.venv/bin/python"
HOSTS=[("local",12),("keunsoo@100.106.141.29",18),("keunsoo@100.75.73.88",18)]


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def state(path,**values):
    previous=json.loads(path.read_text()) if path.exists() else {}
    updated={**previous,**values,"updated_at":now()}
    temporary=path.with_suffix(".tmp");temporary.write_text(json.dumps(updated,indent=2)+"\n");temporary.replace(path)


def call(command):
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError(f"Command failed ({result.returncode}): {result.stdout[-3000:]}")
    return result.stdout


def remote(host,command,*,safe_to_retry=True):
    if host=="local": return call(command)
    # Read-only probes and idempotent directory creation can survive a transient
    # Tailscale/SSH failure. A worker launch is never repeated after an uncertain reply.
    for attempt in range(3 if safe_to_retry else 1):
        result=subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=10",host,shlex.join(command)],
            text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if result.returncode==0: return result.stdout
        if result.returncode!=255 or not safe_to_retry or attempt==2:
            raise RuntimeError(f"Remote command failed ({result.returncode}): {result.stdout[-3000:]}")
        time.sleep(2*(attempt+1))


def require_deadline(stage_name):
    boundary=datetime.date(2026,10,31) if stage_name in {"cost","screen","selection"} else datetime.date(2026,11,28)
    if datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).date()>boundary:
        raise RuntimeError(f"{stage_name} boundary {boundary} reached; preserve incomplete evidence and use the fallback paper path")


def owner_status_code(directory,owner,command):
    """A stale or absent completion record is not permission to rerun an owner."""
    return """import json,os
from pathlib import Path
d=Path(%r); stem='owner-%%02d'%%%d
done=d/(stem+'.done.json'); pidfile=d/(stem+'.pid.json'); ledger=d/(stem+'.sqlite')
if done.exists():
 print(json.dumps({'status':'done',**json.loads(done.read_text())}))
elif pidfile.exists():
 pid=json.loads(pidfile.read_text())['pid']; proc=Path('/proc')/str(pid)/'cmdline'
 actual=proc.read_bytes().rstrip(b'\\0').decode().split('\\0') if proc.exists() else []
 print(json.dumps({'status':'running' if actual==%r else 'interrupted','pid':pid}))
else:
 print(json.dumps({'status':'interrupted' if ledger.exists() else 'new'}))
""" % (str(directory),owner,command)


def worker(args):
    import fcntl
    args.output.mkdir(parents=True,exist_ok=True)
    lock=open(args.output/f"owner-{args.owner:02d}.lock","a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    command=[PYTHON,str(ROOT/"validation/publication_v3/study.py"),"run","--stage",args.stage,
        "--owner",str(args.owner),"--owners","48","--output",str(args.output/f"owner-{args.owner:02d}.sqlite")]
    if args.freeze: command += ["--freeze",str(args.freeze)]
    started=now();result=subprocess.run(command)
    done=dict(owner=args.owner,stage=args.stage,started=started,ended=now(),returncode=result.returncode)
    tmp=args.output/f"owner-{args.owner:02d}.done.tmp";tmp.write_text(json.dumps(done)+"\n")
    tmp.replace(args.output/f"owner-{args.owner:02d}.done.json")
    raise SystemExit(result.returncode)


def confirmation_preflight(stage_name,out,freeze):
    """Wait for additional workers only; verify every host before new dispatch."""
    expected=json.loads(freeze.read_text());directory=out/stage_name
    while True:
        require_deadline(stage_name);snapshots=[];offset=0
        for host,count in HOSTS:
            code="import os,json,hashlib,shutil,sys;from pathlib import Path;sys.path.insert(0,"+repr(str(ROOT/'src'))+");sys.path.insert(0,"+repr(str(ROOT/'validation/publication_v3'))+");from study import hashes,environment;p=Path("+repr(str(directory))+");owners="+repr(list(range(offset,offset+count)))+";m=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());new=sum(not any(p.joinpath('owner-%02d'%i+suffix).exists() for suffix in ('.pid.json','.done.json','.sqlite')) for i in owners);print(json.dumps({'source_hashes':hashes(),'environment':environment(),'load1':os.getloadavg()[0],'cpus':os.cpu_count(),'available_bytes':int(m['MemAvailable'].split()[0])*1024,'free_disk_bytes':shutil.disk_usage("+repr(str(ROOT))+").free,'new_workers':new}))"
            measured=json.loads(remote(host,[PYTHON,'-c',code]));offset+=count
            for key in ('source_hashes','environment'):
                if measured[key]!=expected[key]:raise ValueError('Confirmation host '+key+' differs from immutable seal')
            snapshots.append(dict(host=host,planned_workers=count,**measured))
        state(out/'execution-state.json',phase=stage_name,confirmation_resource_snapshots=snapshots,error=None)
        if all(r['new_workers']==0 or (r['load1']+r['new_workers']<=r['cpus'] and r['available_bytes']>=r['new_workers']*1024**3 and r['free_disk_bytes']>=20*1024**3) for r in snapshots):return
        state(out/'execution-state.json',phase=stage_name,status='waiting_for_confirmation_resources');time.sleep(30)


def stage(stage_name,out,freeze=None):
    require_deadline(stage_name)
    directory=out/stage_name;directory.mkdir(parents=True,exist_ok=True)
    if freeze:confirmation_preflight(stage_name,out,freeze)
    jobs=[dict(host=host,owner=owner) for owner,host in enumerate(host for host,count in HOSTS for unused in range(count))]
    ownership=directory/"ownership.json"
    if ownership.exists() and json.loads(ownership.read_text())!=jobs: raise ValueError("Fixed owner assignment changed")
    if not ownership.exists(): ownership.write_text(json.dumps(jobs,indent=2)+"\n")
    if freeze:
        for host,unused in HOSTS:
            if host!="local":
                remote(host,["mkdir","-p",str(freeze.parent)])
                call(["rsync","-a",str(freeze),host+":"+str(freeze)])
    owner=0
    for host,count in HOSTS:
        remote(host,["mkdir","-p",str(directory)])
        for unused in range(count):
            done=directory/f"owner-{owner:02d}.done.json"
            command=[PYTHON,str(ROOT/"validation/publication_v3/execute.py"),"worker","--stage",stage_name,"--owner",str(owner),"--output",str(directory)]
            if freeze: command += ["--freeze",str(freeze)]
            # A completed owner is never relaunched. Failed owners stop the stage.
            existing=json.loads(remote(host,["python3","-c",owner_status_code(directory,owner,command)]))
            if existing["status"]=="interrupted": raise RuntimeError(f"Owner {owner} interrupted; inspect preserved ledger/logs before any resume")
            if existing["status"]=="done" and existing["returncode"]: raise RuntimeError(f"Owner {owner} failed; no automatic replacement")
            if existing["status"]=="new":
                launch="import os,subprocess,json; from pathlib import Path; p="+repr(str(directory/f"owner-{owner:02d}.log"))+"; f=open(p,'ab'); q=subprocess.Popen("+repr(command)+",cwd="+repr(str(ROOT))+",env={**os.environ,'PYTHONPATH':"+repr(str(ROOT/"src"))+",'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True); Path("+repr(str(directory/f"owner-{owner:02d}.pid.json"))+").write_text(json.dumps({'pid':q.pid,'command':"+repr(command)+"})+'\\n');print(json.dumps({'pid':q.pid}))"
                remote(host,["python3","-c",launch],safe_to_retry=False)
            owner+=1
    while True:
        completed=0
        for host,count in HOSTS:
            read="import json;from pathlib import Path;p=Path("+repr(str(directory))+ "); stale=[f.name for f in p.glob('owner-*.pid.json') if not p.joinpath(f.name.replace('.pid.json','.done.json')).exists() and not Path('/proc',str(json.loads(f.read_text())['pid']),'cmdline').exists()]; assert not stale, 'Interrupted owners: '+str(stale); print(json.dumps([json.loads(f.read_text()) for f in p.glob('owner-*.done.json')]))"
            results=json.loads(remote(host,["python3","-c",read]))
            if any(item["returncode"] for item in results): raise RuntimeError("Fixed owner failed; inspect retained logs. No automatic replacement.")
            completed+=len(results)
        state(out/"execution-state.json",phase=stage_name,completed_owners=completed,planned_owners=48,status="running")
        if completed==48: break
        require_deadline(stage_name)
        time.sleep(30)
    collected=directory/"collected";collected.mkdir(exist_ok=True)
    for job in jobs:
        source=str(directory/f"owner-{job['owner']:02d}.sqlite")
        if job["host"]=="local":
            with sqlite3.connect(source) as db,sqlite3.connect(collected/Path(source).name) as target: db.backup(target)
        else: call(["rsync","-a",job["host"]+":"+source,str(collected)+"/"])
    summary=directory/"summary.json"
    if not summary.exists(): call([PYTHON,str(ROOT/"validation/publication_v3/study.py"),"summarize",*[str(p) for p in sorted(collected.glob("*.sqlite"))],"--output",str(summary)])
    if json.loads(summary.read_text()).get("complete") is not True: raise RuntimeError("Incomplete stage summary")
    return summary


def pipeline(args):
    args.output.mkdir(parents=True,exist_ok=True);status=args.output/"execution-state.json"
    state(status,status="waiting_for_cost",phase="cost",confirmation_started=False,deadline="2026-12-26",error=None)
    # Existing cost owners were launched before this controller. Read only their ledgers.
    while True:
        require_deadline("cost")
        paths=sorted(args.cost.glob("owner-*.sqlite"));count=0
        for path in paths:
            with sqlite3.connect(path) as db: count+=db.execute("SELECT count(*) FROM replicates").fetchone()[0]
        state(status,cost_completed=count,cost_planned=80)
        if count==80: break
        if count>80: raise ValueError("Unexpected cost-study records")
        time.sleep(30)
    cost_summary=args.cost/"summary.json"
    if not cost_summary.exists(): call([PYTHON,str(ROOT/"validation/publication_v3/study.py"),"summarize",*[str(p) for p in paths],"--output",str(cost_summary)])
    cost_decision=args.cost/"decision.json"
    if not cost_decision.exists(): call([PYTHON,str(ROOT/"validation/publication_v3/control.py"),"cost",str(cost_summary),"--output",str(cost_decision)])
    cost=json.loads(cost_decision.read_text())
    reference=json.loads(args.reference.read_text())
    if reference.get("passed") is not True: raise ValueError("Independent R reference did not pass")
    if cost["status"]!="feasible":
        state(status,status="v2_fallback_resource_limit",cost=cost,confirmation_started=False);return
    from study import environment
    expected=environment()
    if reference.get("environment")!=expected: raise ValueError("R reference numerical environment differs")
    for host,unused in HOSTS:
        code="import sys,json;sys.path.insert(0,"+repr(str(ROOT/"validation/publication_v3"))+ ");sys.path.insert(0,"+repr(str(ROOT/"src"))+ ");from study import environment;print(json.dumps(environment()))"
        if json.loads(remote(host,[PYTHON,"-c",code]))!=expected: raise ValueError("Worker numerical environments differ")
    while True:
        require_deadline("screen");snapshots=[]
        for host,count in HOSTS:
            code="import os,json,shutil;from pathlib import Path;m=dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines());print(json.dumps({'load1':os.getloadavg()[0],'cpus':os.cpu_count(),'available_bytes':int(m['MemAvailable'].split()[0])*1024,'free_disk_bytes':shutil.disk_usage("+repr(str(ROOT))+").free}))"
            measured=json.loads(remote(host,["python3","-c",code]));snapshots.append(dict(host=host,workers=count,**measured))
        state(status,resource_snapshots=snapshots)
        if all(r["load1"]+r["workers"]<=r["cpus"] and r["available_bytes"]>=r["workers"]*1024**3 and r["free_disk_bytes"]>=20*1024**3 for r in snapshots): break
        state(status,status="waiting_for_resources",phase="screen");time.sleep(30)
    screen=stage("screen",args.output)
    selection_summary=stage("selection",args.output)
    decision=args.output/"selection-decision.json"
    if not decision.exists(): call([PYTHON,str(ROOT/"validation/publication_v3/control.py"),"select",str(selection_summary),"--output",str(decision)])
    selected=json.loads(decision.read_text())
    if not selected["selection_eligible"]:
        state(status,status="v2_fallback_no_eligible_candidate",phase="selection",confirmation_started=False,selection=selected);return
    # Freeze/push require a reviewed source commit. Stop here for the next agent
    # pass to bind fresh R, full CI and selection provenance before confirmation.
    state(status,status="candidate_selected_requires_source_seal",phase="selection",selection=selected,
        confirmation_started=False,next_command="Commit the one selected policy, verify R/CI, seal, then dispatch confirmation.")


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    w=sub.add_parser("worker");w.add_argument("--stage",required=True);w.add_argument("--owner",type=int,required=True);w.add_argument("--output",type=Path,required=True);w.add_argument("--freeze",type=Path)
    r=sub.add_parser("pipeline");r.add_argument("--output",type=Path,required=True);r.add_argument("--cost",type=Path,required=True);r.add_argument("--reference",type=Path,required=True)
    d=sub.add_parser("dispatch");d.add_argument("--stage",choices=["main","extension","large","stress"],required=True);d.add_argument("--output",type=Path,required=True);d.add_argument("--freeze",type=Path,required=True)
    args=p.parse_args()
    if args.command=="worker": worker(args)
    else:
        import fcntl
        args.output.mkdir(parents=True,exist_ok=True)
        manager_lock=open(args.output/"manager.lock","a")
        fcntl.flock(manager_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            if args.command=="pipeline": pipeline(args)
            else: stage(args.stage,args.output,args.freeze)
        except Exception as exc:
            state(args.output/"execution-state.json",status="stopped_with_preserved_evidence",error=type(exc).__name__+": "+str(exc));raise


if __name__=="__main__": main()
