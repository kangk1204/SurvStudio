import csv

from fastapi.testclient import TestClient
import pytest

from survival_toolkit.app import app
from test_guarded_marker_inference import cohort
from test_browser_e2e import browser_server, _launch_browser, _wait_for_workspace


def test_experimental_v3_api_export_keeps_nulls_and_raw_calculations():
    with TestClient(app,base_url="http://127.0.0.1") as client:
        dataset=client.post("/api/upload",files={"file":("synthetic.csv",cohort(n=90).to_csv(index=False),"text/csv")}).json()
        response=client.post("/api/marker-evaluation",json={"dataset_id":dataset["dataset_id"],
            "time_column":"time","event_column":"event","marker_columns":["m0","m1","m2"],
            "clinical_columns":["z"],"n_permutations":19,"n_resamples":0,"random_seed":42,
            "diagnostic_policy":"v3_joint_bootstrap"})
        assert response.status_code==200,response.text
        payload=response.json();inference=payload["analysis"]["inference"]
        assert payload["analysis"]["method_version"]=="marker-inference/3"
        assert not inference["allowed"] and inference["engineering_qualification"]["status"]=="not_evaluated"
        assert inference["bootstrap"]["draws"]==9999
        assert all(t.get("p_maxT") is not None for t in inference["residual_tests"] if t["status"]=="calculated")
        assert all(row["Family-wise P"] is None and row["Exploratory raw family-wise P"] is not None
                   and row["Diagnostic policy"]=="v3_joint_bootstrap" for row in payload["display_table"])
        assert payload["analysis"]["locked_recipe"]["inference"]==inference
        assert "withheld" in payload["report"]["results"]


def test_browser_v3_policy_currency_diagnostics_and_csv(browser_server,tmp_path):
    playwright=pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as api:
        browser=_launch_browser(api);page=browser.new_page(viewport={"width":1440,"height":1200})
        page.goto(browser_server,wait_until="networkidle")
        upload=tmp_path/"synthetic.csv";cohort(n=90).to_csv(upload,index=False)
        page.locator("#datasetFile").set_input_files(str(upload));_wait_for_workspace(page)
        page.locator('[data-tab="markers"]').click()
        assert page.locator("#markerDiagnosticPolicy").input_value()=="v2_holm"
        for checklist in ("#markerChecklist","#markerClinicalChecklist"):
            for box in page.locator(checklist+" input[type=checkbox]").all(): box.set_checked(False)
        for marker in ("m0","m1","m2"): page.locator(f"#markerChecklist input[value='{marker}']").check()
        page.locator("#markerClinicalChecklist input[value='z']").check()
        page.locator("#markerDiagnosticPolicy").select_option("v3_joint_bootstrap")
        page.locator("#panel-markers .options-details > summary").click()
        page.locator("#markerPermutations").fill("19");page.locator("#markerResamples").fill("0")
        page.locator("#markerRandomSeed").fill("42");page.locator("#runMarkersButton").click()
        page.wait_for_function("!document.getElementById('downloadMarkersCsvButton').disabled",timeout=120000)
        assert "9999 draws" in page.locator("#markersInferenceNote").text_content()
        assert "Monte Carlo 95% interval" in page.locator("#markersInferenceNote").text_content()
        page.locator("#markersDiagnostics > summary").click()
        assert "joint bootstrap p" in page.locator("#markersDiagnosticsTable").inner_text().lower()
        page.locator("#markerDiagnosticPolicy").select_option("v2_holm")
        page.wait_for_function("document.querySelector('[data-run-status=markers]').textContent === 'Settings changed'")
        page.locator("#markerDiagnosticPolicy").select_option("v3_joint_bootstrap")
        page.wait_for_function("document.querySelector('[data-run-status=markers]').textContent === 'Up to date'")
        page.locator("#panel-markers .export-menu > summary").click()
        with page.expect_download() as downloaded: page.locator("#downloadMarkersCsvButton").click()
        destination=tmp_path/"v3-withheld.csv";downloaded.value.save_as(destination)
        rows=list(csv.reader(destination.open(encoding="utf-8-sig")))
        header=next(i for i,row in enumerate(rows) if row and row[0]=="Marker")
        parsed=[dict(zip(rows[header],row)) for row in rows[header+1:]]
        assert parsed and all(row["Family-wise P"] in ("","NA") and row["Diagnostic policy"]=="v3_joint_bootstrap" for row in parsed)
        page.screenshot(path=str(tmp_path/"v3-diagnostic-screen.png"),full_page=True)
        browser.close()
