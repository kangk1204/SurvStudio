"""Build compact public aggregate tables from preserved private evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def table(path,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,allow_nan=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def run(evidence,v2,output):
    output.mkdir(parents=True,exist_ok=False)
    paths={'v2':v2/'confirmation-v2-summary.json','v3_main':evidence/'evidence-v3/main-summary.json',
        'v3_extension':evidence/'extension-audit-r1/audit.json','public_case':evidence/'case-evaluation/external-aggregate.json',
        'common_benchmark':evidence/'common-benchmark/verification.json','fresh_environment':evidence/'fresh-environment-verification.json',
        'state_surfaces':evidence/'comparison-surfaces-r2/verification.json','R_eight_tasks':evidence/'eight-task-reference/verification.json',
        'native_probes':evidence/'native-probes-r1/verification.json','native_followups':evidence/'native-followups-r1/verification.json',
        'R_public_case':evidence/'case-independent-R/verification.json','R_v3_aggregation':evidence/'evidence-v3/main-R-verification.json'}
    records={k:read(p) for k,p in paths.items()}
    for name in ('v2','v3_main'):table(output/(name+'.csv'),records[name]['summaries'])
    table(output/'v3_extension_bounds.csv',records['v3_extension']['summaries'])
    table(output/'invalid_original_records.csv',records['v3_extension']['invalid_records'])
    old=v2/'review_package/SurvStudio_v2_review_package_20261003/aggregate_evidence'
    for name in ('power-retention-uncertainty.csv','power-retention-independent-R.csv'):
        shutil.copyfile(old/name,output/name)
    shutil.copyfile(old/'cases-v2/pooled-case-results.csv',output/'existing_external_cases.csv')
    cases=[];calibration=[]
    for result in records['public_case']['results']:
        for h in result['horizons']:
            cases.append(dict(case=result['case'],horizon_years=h['horizon_days']/365.25,**h['metrics'],
                inference_status=h['inference']['status'],reasons=h['inference']['reasons'],
                cohort_n=sum(g['n'] for g in h['model_calibration']['clinical_plus_PGR']['fixed_group_calibration']),
                known_benchmark=True,recipe_hash=result['recipe_hash']))
            for model,v in h['model_calibration'].items():
                for g in v['fixed_group_calibration']:calibration.append(dict(case=result['case'],horizon_years=h['horizon_days']/365.25,model=model,**g))
    table(output/'public_external_case.csv',cases);table(output/'public_fixed_group_calibration.csv',calibration)
    table(output/'benchmark.csv',records['common_benchmark']['rows'])
    numeric=[]
    for group,key in [('public_case','R_public_case'),('v3_aggregation','R_v3_aggregation')]:
        for row in records[key]['checks']:numeric.append(dict(validation_group=group,**row))
    for group,path in [('eight_task',evidence/'eight-task-reference/numerical-comparison.csv'),('native_lifelines',evidence/'native-probes-r1/independent-numeric-comparison.csv')]:
        for row in csv.DictReader(path.open()):numeric.append(dict(validation_group=group,**row))
    table(output/'independent_numeric_checks.csv',numeric)
    task_records=[]
    for tool in ('SurvStudio','R survival/Hmisc/rms'):
        for row in records['R_eight_tasks']['tasks']:
            task_records.append(dict(tool=tool,task=row['task'],status=row['SurvStudio_and_R_status'],
                display_code='N' if tool=='SurvStudio' or row['task'] in ('event_coding','KM_estimation','adjusted_HR_reference') else 'S',
                capability_layer='native_application' if tool=='SurvStudio' else 'independent_added_R_workflow',detail=row['execution_detail']))
    for row in csv.DictReader((evidence/'native-probes-r1/task-records.csv').open()):
        row=dict(row)
        if row['tool']=='lifelines' and row['task'] in ('selection_internal_evaluation','locked_external_validation'):
            row.update(status='executed_added_script',capability_layer='added_script',detail='Fixed training-only partial-likelihood selection or native-fitter pickle persistence; estimand differences retained.')
        code='N' if row['status']=='executed_numeric_match' or row['tool']=='mlsurv' and row['task']=='locked_external_validation' else 'S' if row['status']=='executed_added_script' else 'D'
        if row['tool']=='mlsurv' and row['task']=='adjusted_HR_reference':
            code='N';row.update(status='executed_common_model_numeric_match',detail='Native CoxPHModel coefficients match independent R on the untied common fixture. Full-learner significance tests are not compared.')
        row['display_code']=code;task_records.append(row)
    tasks=[r['task'] for r in records['R_eight_tasks']['tasks']]
    for tool,code,status,detail in [('surviveR','P','authentication_pending','Official service opened; user authentication required. No account created or analysis submitted.'),
                                  ('KM Plotter','L','access_limited_execution_unverified','Non-human execution requires operator permission under terms. No analysis submitted; capability not inferred from nonexecution.')]:
        for task in tasks:task_records.append(dict(tool=tool,task=task,display_code=code,status=status,capability_layer='not_executed',detail=detail))
    table(output/'tool_tasks.csv',task_records)
    table(output/'state_checks.csv',records['state_surfaces']['checks'])
    shutil.copyfile(evidence/'comparison-surfaces-r2/ablation.csv',output/'controlled_ablation.csv')
    shutil.copyfile(evidence/'comparison-inputs-r2/contract.json',output/'comparison-contract.json')
    shutil.copytree(evidence/'comparison-inputs-r2',output/'synthetic_inputs')
    shutil.copyfile(evidence/'python-lock.txt',output/'python-lock.txt')
    for name in ('fresh_environment','state_surfaces','native_followups'):
        (output/(name+'-verification.json')).write_text(json.dumps(records[name],indent=2,allow_nan=False)+'\n')
    extension=records['v3_extension']
    completeness=[dict(stage='v3_main',planned=120000,validated_paired_rows=120000,unusable=0,absent=0,status='complete'),
        dict(stage='v3_extension',planned=12000,validated_paired_rows=extension['valid_paired_rows'],unusable=extension['invalid_paired_rows'],absent=extension['absent_rows'],status=extension['status']),
        dict(stage='v3_large',planned=1500,validated_paired_rows=0,unusable=0,absent=1500,status='never_dispatched'),
        dict(stage='v3_stress',planned=16000,validated_paired_rows=0,unusable=0,absent=16000,status='never_dispatched')]
    table(output/'v3_completeness.csv',completeness)
    summary=dict(v2_datasets=records['v2']['planned_confirmation_datasets'],v2_method_records=records['v2']['planned_confirmation_datasets']*3,
        v3_main_datasets=120000,v3_main_method_records=600000,v3_total_planned=149500,
        v3_extension_valid=extension['valid_paired_rows'],v3_extension_unusable=extension['invalid_paired_rows'],v3_extension_absent=extension['absent_rows'],
        v3_adoption='NO_GO',fresh_reproduction=records['fresh_environment']['passed'],state_surface_checks=len(records['state_surfaces']['checks']),
        R_public_case_checks=len(records['R_public_case']['checks']),R_main_aggregate_checks=len(records['R_v3_aggregation']['checks']),
        web_comparison_status='surviveR authentication pending; KM Plotter permission absent',
        patient_rows_or_actual_recipes_in_public_bundle=False,submission_ready=False,
        missing_author_facts=['authors','affiliations','corresponding author','contributions','ethics determination','data-use basis','funding','competing interests'],
        source_input_hashes={k:sha(p) for k,p in paths.items()},assembler_sha256=sha(Path(__file__)))
    (output/'evidence-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'aggregate-manifest.json').write_text(json.dumps({str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()},indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--v2',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.evidence,a.v2,a.output)
