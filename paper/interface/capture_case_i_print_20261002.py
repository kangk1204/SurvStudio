"""Render a saved, verified case-I analysis with the current UI and actual attached matrix.

The analysis is computed by paper/script 01; this records a rendering of that saved result,
not another numerical run or an edited screenshot.
"""
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import survival_toolkit.reporting as reporting_module
import survival_toolkit.plots as plots_module
from playwright.sync_api import sync_playwright
from survival_toolkit.app import _marker_display_rows, _trim_marker_table
from survival_toolkit.plots import build_marker_summary_figure, build_marker_rank_figure, build_marker_stability_figure
from survival_toolkit.reporting import remark_checklist

repository = Path(sys.argv[1]); output = Path(sys.argv[2]); output.mkdir(parents=True,exist_ok=True)
source = repository / "paper/results/tcga_analysis.json"
stamp = json.loads((source.parent/"stamps/tcga_analysis.json.json").read_text())
assert stamp["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
assert stamp["survstudio"]["commit"] == "3e0c4af1"
analysis = json.loads(source.read_text())
assert analysis["cohort"]["n"] == 484 and analysis["cohort"]["events"] == 177
payload = {"analysis":_trim_marker_table(analysis),"display_table":_marker_display_rows(analysis),
    "summary_figure":build_marker_summary_figure(analysis),"stability_figure":build_marker_stability_figure(analysis),
    "rank_figure":build_marker_rank_figure(analysis),"report":remark_checklist(analysis)}
with socket.socket() as sock:
    sock.bind(("127.0.0.1",0)); port=sock.getsockname()[1]
url=f"http://127.0.0.1:{port}"
with (output/"server.log").open("w") as log:
    server=subprocess.Popen([sys.executable,"-m","survival_toolkit","serve","--port",str(port)],stdout=log,stderr=log)
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(url+"/api/health",timeout=2).close(); break
            except OSError: time.sleep(.2)
        with sync_playwright() as api:
            browser=api.chromium.launch(headless=True)
            page=browser.new_page(viewport={"width":int(sys.argv[3]) if len(sys.argv)>3 else 900,"height":1200},device_scale_factor=3)
            page.goto(url,wait_until="networkidle")
            page.locator("#loadTcgaUploadReadyButton").click()
            page.locator("#workspace").wait_for(state="visible")
            page.wait_for_function("Boolean(state.dataset?.dataset_id) && refs.markerMatrixIdColumn.options.length > 0")
            page.locator('[data-tab="markers"]').click()
            page.locator("#markerMatrixDetails > summary").click()
            page.locator("#markerMatrixFile").set_input_files(str(repository/"paper/data/cohorts/luad/TCGA-LUAD/raw/HiSeqV2.gz"))
            page.locator("#attachMarkerMatrixButton").click()
            page.wait_for_function("Boolean(state.markerMatrix?.matrix_id)",timeout=240000)
            page.evaluate("setCheckedValues(refs.markerClinicalChecklist,['age','sex','stage_group']); renderMarkerSelectionLine();")
            config=page.evaluate("({...currentBaseConfig(),...markerRequestFields()})")
            assert config["time_column"]=="os_months" and config["event_column"]=="os_event"
            assert set(config["clinical_columns"])=={"age","sex","stage_group"}
            assert config["n_permutations"]==1000 and config["n_resamples"]==200 and config["random_seed"]==20260926
            payload["request_config"]=config
            payload["dataset_hash"]=page.evaluate("state.dataset.dataset_hash")
            payload["marker_matrix"]=page.evaluate("state.markerMatrix")
            page.evaluate("async p => {state.markers=p; await renderMarkerResults(p); syncDownloadButtonAvailability();}",payload)
            page.locator("#markersSummaryPlot .main-svg").first.wait_for(state="visible")
            page.locator("#markersInsightBoard").scroll_into_view_if_needed()
            page.locator("#markersInsightBoard .insight-details > summary").click()
            text=page.locator("#panel-markers").inner_text()
            assert "gap-adjusted" in text and "0.650" in text
            assert "the whole selection procedure reached" in text
            assert "the selected-marker model reached" not in text
            for selector, name in [("#markersInsightBoard", "case_i_insight.png"), ("#markersSummaryPlot", "case_i_summary.png")]:
                page.evaluate("selector => {const r=document.querySelector(selector).getBoundingClientRect(); window.scrollTo(0, window.scrollY+r.top-100);}", selector)
                page.locator(selector).screenshot(path=str(output/name))
            page.locator("#markersInsightBoard .insight-details > summary").click()
            page.evaluate("() => {const r=document.querySelector('#markersSummaryPlot').getBoundingClientRect(); window.scrollTo(0, window.scrollY+r.top-100);}")
            page.screenshot(path=str(output/"markers_tab_case_study_i.png"))
            (output/"provenance.json").write_text(json.dumps({"mode":"rendering of saved paper analysis with a freshly attached real matrix",
                "analysis_sha256":stamp["sha256"],"analysis_stamp":stamp,
                "summary_png_sha256":hashlib.sha256((output/"case_i_summary.png").read_bytes()).hexdigest(),
                "capture_helper_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"request_config":config,
                "rendering_repository":str(Path(reporting_module.__file__).resolve().parents[2]),
                "rendering_commit":subprocess.check_output(["git","-C",str(Path(reporting_module.__file__).resolve().parents[2]),"rev-parse","HEAD"],text=True).strip(),
                "rendering_source_sha256":{
                    str(p.relative_to(Path(reporting_module.__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(reporting_module.__file__).resolve(),Path(plots_module.__file__).resolve(),Path(reporting_module.__file__).resolve().parent/"static/app_markers.js"]
                },
                "matrix":payload["marker_matrix"],"visible_text":text,"viewport":[int(sys.argv[3]) if len(sys.argv)>3 else 900,1200],"device_scale":3,
                "summary_css_width":page.locator("#markersSummaryPlot").evaluate("e=>e.getBoundingClientRect().width"),
                "summary_min_svg_font_px":page.locator("#markersSummaryPlot").evaluate("e=>Math.min(...Array.from(e.querySelectorAll(\"svg text\")).filter(t=>t.textContent.trim()).map(t=>parseFloat(getComputedStyle(t).fontSize)))") },indent=2)+"\n")
            browser.close()
    finally:
        server.terminate();server.wait(timeout=20)
