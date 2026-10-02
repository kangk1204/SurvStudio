"""Durable, fixed-ownership process launcher; start under nohup on each assigned server.

Resume runs only unfinished indexes, retaining every recorded failure. All workers use
the same total/modulo ownership; the inventory must be reviewed before dispatch.
"""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from study import PROTOCOL, verify_freeze


def launch(a):
    freeze_hash=verify_freeze(a.freeze)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.offset<0 or a.slots<1 or a.offset+a.slots>a.total_workers:
        raise ValueError("Invalid ownership allocation")
    protocol=json.loads(PROTOCOL.read_text())
    cells=[dict(stage="main",condition=c,**protocol["main"]) for c in protocol["conditions"]]
    cells += [dict(stage="extension",condition=c,**e) for e in protocol["extensions"] for c in protocol["conditions"]]
    cells += [dict(stage="large",condition=c,n=180,p=3000,replicates_per_condition=500) for c in protocol["large_marker"]["conditions"]]
    inventory=dict(host=socket.gethostname(),total_workers=a.total_workers,
                   workers=list(range(a.offset,a.offset+a.slots)),freeze_hash=freeze_hash,cells=cells)
    inventory_path=out/"ownership.json"
    if inventory_path.exists() and json.loads(inventory_path.read_text())!=inventory:
        raise ValueError("Existing ownership cannot be silently reassigned")
    inventory_path.write_text(json.dumps(inventory,indent=2)+"\n")
    env=dict(os.environ,OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    def worker(worker_id):
        status_path=out/f"worker-{worker_id}-status.json"
        for cell_index,cell in enumerate(cells):
            name=f"{cell['stage']}-{cell['condition']}-n{cell['n']}-p{cell['p']}-worker{worker_id}"
            ledger=out/(name+".sqlite")
            cmd=[sys.executable,str(Path(__file__).with_name("study.py")),"run","--stage",cell["stage"],
                 "--condition",cell["condition"],"--n",str(cell["n"]),"--p",str(cell["p"]),
                 "--stop",str(cell["replicates_per_condition"]),"--workers",str(a.total_workers),
                 "--worker",str(worker_id),"--output",str(ledger),"--freeze",a.freeze]
            state=dict(worker=worker_id,cell=cell,cell_index=cell_index,command=cmd,status="running",updated=time.time())
            status_path.write_text(json.dumps(state,indent=2)+"\n")
            with (out/(name+".log")).open("a") as log:
                result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            state.update(status="completed" if result.returncode==0 else "process_failed",returncode=result.returncode,updated=time.time())
            status_path.write_text(json.dumps(state,indent=2)+"\n")
            if result.returncode:
                raise RuntimeError(f"Worker {worker_id} process failed for {name}; ledger preserved")
        state.update(status="all_assigned_cells_completed",updated=time.time())
        status_path.write_text(json.dumps(state,indent=2)+"\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.slots) as pool:
        list(pool.map(worker,range(a.offset,a.offset+a.slots)))
    (out/"server-complete.json").write_text(json.dumps({**inventory,"completed":time.time()},indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--freeze",required=True);p.add_argument("--output",required=True)
    p.add_argument("--total-workers",type=int,required=True);p.add_argument("--offset",type=int,required=True);p.add_argument("--slots",type=int,required=True)
    launch(p.parse_args())
