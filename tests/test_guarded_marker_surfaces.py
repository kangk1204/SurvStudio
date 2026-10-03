import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from survival_toolkit.app import app
from test_guarded_marker_inference import cohort
from test_browser_e2e import browser_server, _launch_browser, _wait_for_workspace


def test_qualified_linear_nonrejection_remains_assumption_dependent_through_export():
    with TestClient(app,base_url="http://127.0.0.1") as client:
        dataset=client.post("/api/upload",files={"file":("synthetic.csv",cohort().to_csv(index=False),"text/csv")}).json()
        response=client.post("/api/marker-evaluation",json={"dataset_id":dataset["dataset_id"],
            "time_column":"time","event_column":"event","marker_columns":["m0","m1","m2"],
            "clinical_columns":["z"],"clinical_basis":"linear","n_permutations":19,"n_resamples":2})
        assert response.status_code==200,response.text
        payload=response.json()
        inference=payload['analysis']['inference']
        assert inference['allowed'] and inference['status']=='assumption_dependent'
        assert inference['engineering_qualification']['kernel_sources_match'] is True
        assert all(row['Family-wise P'] is not None for row in payload['display_table'])
        assert payload['analysis']['locked_recipe']['inference']==inference
        assert 'not universal' in payload['report']['results']
        assert 'extensions do not establish broader qualification' in payload['report']['results']


def test_api_returns_consistent_withheld_display_report_and_figures():
    with TestClient(app,base_url="http://127.0.0.1") as client:
        dataset=client.post("/api/upload",files={"file":("synthetic.csv",cohort(n=500,nonlinear=True).to_csv(index=False),"text/csv")}).json()
        response=client.post("/api/marker-evaluation",json={"dataset_id":dataset["dataset_id"],
            "time_column":"time","event_column":"event","marker_columns":["m0","m1","m2"],
            "clinical_columns":["z"],"clinical_basis":"linear","n_permutations":19,"n_resamples":2})
        assert response.status_code==200,response.text
        payload=response.json()
        assert payload["analysis"]["inference"]["status"]=="withheld"
        assert all(row["P value"] is None and row["Family-wise P"] is None and row["Inference status"]=="withheld"
                   for row in payload["display_table"])
        assert payload['analysis']['method_version']=='marker-inference/2'
        assert all(row['Engineering qualification']=='passed_supported_conditions_only' and row['Exploratory raw family-wise P'] is not None
                   for row in payload['display_table'])
        assert "withheld" in payload["report"]["results"]
        for name in ("summary_figure","stability_figure","rank_figure"):
            assert "inference withheld" in payload[name]["layout"]["title"]["text"]
        invalid=client.post("/api/marker-evaluation",json={**payload["request_config"],"clinical_basis":"automatic"})
        assert invalid.status_code==422


def test_browser_spline_choice_currency_and_withheld_exports(browser_server,tmp_path):
    playwright=pytest.importorskip("playwright.sync_api")
    with playwright.sync_playwright() as api:
        browser=_launch_browser(api)
        page=browser.new_page(viewport={"width":1440,"height":1200})
        page.goto(browser_server,wait_until="networkidle")
        upload=tmp_path/"synthetic.csv"
        cohort(n=500,nonlinear=True).to_csv(upload,index=False)
        page.locator("#datasetFile").set_input_files(str(upload));_wait_for_workspace(page)
        page.locator('[data-tab="markers"]').click()
        assert page.locator("#markerClinicalBasis").input_value()=="linear"
        for checklist in ("#markerChecklist", "#markerClinicalChecklist"):
            for box in page.locator(checklist+" input[type=checkbox]").all(): box.set_checked(False)
        for name in ("m0","m1","m2"):
            page.locator(f"#markerChecklist input[value='{name}']").check()
        page.locator("#markerClinicalChecklist input[value='z']").check()
        page.locator("#panel-markers .options-details > summary").click()
        page.locator("#markerPermutations").fill("19");page.locator("#markerResamples").fill("2")
        page.locator("#markerRandomSeed").fill("42")
        page.locator("#runMarkersButton").click()
        page.wait_for_function("!document.getElementById('downloadMarkersCsvButton').disabled",timeout=120000)
        page.locator("#markerClinicalBasis").select_option("restricted_cubic_spline")
        page.wait_for_function("document.querySelector('[data-run-status=markers]').textContent === 'Settings changed'")
        page.locator("#markerClinicalBasis").select_option("linear")
        page.wait_for_function("document.querySelector('[data-run-status=markers]').textContent === 'Up to date'")
        assert "inference is withheld" in page.locator("#markersInsightBoard").inner_text()
        assert "inference status" in page.locator("#markersTableShell").inner_text().lower()
        assert "assumptions" in page.locator("#markersInferenceNote").text_content()
        assert page.locator("#markersTableShell").get_by_text("withheld",exact=True).count()>0
        page.locator("#markersDiagnostics > summary").click()
        assert "holm p" in page.locator("#markersDiagnosticsTable").inner_text().lower()
        image=tmp_path/"withheld-marker-screen.png"
        page.screenshot(path=str(image),full_page=True)
        assert image.stat().st_size>0
        page.locator("#panel-markers .export-menu > summary").click()
        with page.expect_download() as downloaded:
            page.locator("#downloadMarkersCsvButton").click()
        destination=tmp_path/"withheld-markers.csv";downloaded.value.save_as(destination)
        import csv
        parsed=list(csv.reader(destination.open(encoding="utf-8-sig")))
        header=next(i for i,row in enumerate(parsed) if row and row[0]=="Marker")
        rows= [dict(zip(parsed[header],row)) for row in parsed[header+1:]]
        assert all(row["Family-wise P"] in ("", "NA") and row["Inference status"]=="withheld" for row in rows)
        browser.close()
