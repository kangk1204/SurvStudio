"""Export an explicit aggregate allowlist from completed private case evidence.

Patient-level inputs, scores, full marker tables and full recipes are never copied.
The original case outputs and independent reference hashes remain unchanged.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
from reanalyse_cases import save


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(evidence, output):
    from survival_toolkit.marker_qualification import qualify_marker_result
    evidence=Path(evidence).resolve();out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    results=[];pooled_rows=[];quantities=0;maximum=0.
    for case in ('I','IV','V'):
        for basis in ('linear','restricted_cubic_spline'):
            key=f'case-{case}-{basis}';development=evidence/'cases'/key;external=evidence/'cases_final'/key
            reference=json.loads((external/'independent-R-verification.json').read_text())
            if not reference['passed'] or reference['unavailable']:
                raise ValueError('Independent case reference did not pass: '+key)
            for name,expected in reference['hashes'].items():
                if digest(external/name)!=expected: raise ValueError('Private reference input changed: '+key+'/'+name)
            raw=json.loads((development/'development-analysis.json').read_text())
            result=qualify_marker_result(raw);inference=result['inference'];original=raw['inference']
            configuration=json.loads((external/'configuration.json').read_text())
            if digest(development/'development-analysis.json')!=configuration['development_sha256']:
                raise ValueError('Development evidence changed: '+key)
            recipe=json.loads((external/'recipe.json').read_text())
            if recipe['recipe_hash']!=result['locked_recipe']['recipe_hash']:
                raise ValueError('Qualified fixed recipe differs: '+key)
            residual=original.get('residual_tests',[])
            residual_counts=Counter((r['diagnostic'],r['status'],bool(r.get('p_holm') is not None and r['p_holm']<.01)) for r in residual)
            item={'case':case,'clinical_basis':basis,'existing_external_data_reanalysis':True,
                  'development':{k:raw['cohort'][k] for k in ('n','events','n_markers_evaluated')},
                  'settings':raw['settings'],'original_inference_status':original['status'],
                  'qualified_inference_status':inference['status'],'qualified_allowed':inference['allowed'],
                  'engineering_qualification':inference.get('engineering_qualification'),
                  'reason_counts':dict(Counter(r.split(':',1)[0] for r in inference['reasons'])),
                  'clinical_tests':original.get('clinical_tests',[]),
                  'residual_summary':[{'diagnostic':k[0],'status':k[1],'holm_rejected':k[2],'markers':v} for k,v in sorted(residual_counts.items())],
                  'signature_markers':raw['signature']['markers'],
                  'exploratory_signature_apparent_c':raw['signature']['apparent_c'],
                  'signature_gain_left_out':raw['signature']['signature_gain_left_out'],
                  'signature_gain_left_out_ci':raw['signature']['signature_gain_left_out_ci'],
                  'n_signature_replicates':raw['signature']['n_signature_replicates'],
                  'optimism_corrected_c':result['signature']['optimism_corrected_c'],
                  'fixed_recipe_hash':recipe['recipe_hash'],'external_configuration':configuration,
                  'development_configuration':json.loads((development/'configuration.json').read_text()),
                  'independent_R':{k:reference[k] for k in ('status','passed','quantities','unavailable','maximum_absolute_difference','reference_script_sha256')},
                  'private_evidence_hashes':{name:digest(external/name) for name in
                       ('recipe.json','complete.json','eligibility.json','pooled-aggregate.json','independent-R-verification.json')}}
            destination=out/key;destination.mkdir(exist_ok=True)
            for name in ('external-aggregate.csv','pooled-aggregate.json','configuration.json','eligibility.json',
                         'independent-R-verification.json','independent-R-comparison.csv',
                         'independent-R-pooling.csv','independent-R-values.csv','R-session.txt'):
                shutil.copyfile(external/name,destination/name)
            pooled=json.loads((external/'pooled-aggregate.json').read_text())
            for group,data in pooled.items():
                for metric,metric_data in data['quantities'].items():
                    p=metric_data['pooled']
                    pooled_rows.append({'case':case,'clinical_basis':basis,'endpoint':data['endpoint'],
                        'scaling':group.split(':')[1],'role':data['role'],'n':data['n'],'events':data['events'],
                        'cohorts':data['k'],'metric':metric,'eligible_cohorts':metric_data['eligible_cohorts'],
                        'unavailable_cohorts':';'.join(metric_data['unavailable_cohorts']),
                        'inference_status':inference['status'],**(p or {})})
            results.append(item);quantities+=reference['quantities'];maximum=max(maximum,reference['maximum_absolute_difference'])
    pd.DataFrame(pooled_rows).to_csv(out/'pooled-case-results.csv',index=False)
    summary={'existing_external_data_reanalysis':True,'primary_basis':'linear',
             'spline_role':'prespecified sensitivity','primary_scaling':'as_measured',
             'primary_case_V_endpoint':'rfs','secondary_case_V_endpoint':'dmfs',
             'external_basis_or_signature_selection':False,'patient_level_data_included':False,
             'independent_R_quantities':quantities,'independent_R_passed':True,
             'maximum_absolute_difference':maximum,'analyses':results,
             'collector_script_sha256':digest(Path(__file__))}
    save(out/'case-summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('analyses',)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.evidence,a.output)
