"""Product release gate, separate from the frozen research numerical kernel.

Failed or unevaluated engineering profiles are exploratory even when a particular
dataset's diagnostics did not reject. Never enable a profile using favorable subsets.
"""
import copy
import hashlib
import json
from pathlib import Path

REGISTRY=Path(__file__).with_name('data')/'marker_qualification.json'
KERNEL_FILES={f'src/survival_toolkit/{name}.py' for name in
              ('clinical_basis','marker_diagnostics','marker_evaluation','marker_screen','encoding','analysis')}


def qualification(basis,method_version):
    try:
        raw=REGISTRY.read_bytes();registry=json.loads(raw)
        study=registry.get('studies',{}).get(method_version,registry)
        profile=study['profiles'][basis]
        evaluated=study.get('study_complete') is True and study.get('method_version')==method_version
        source_bound=True
        sources_match=None
        if profile['status']=='passed_supported_conditions_only':
            sources=study.get('kernel_source_hashes')
            source_bound=isinstance(sources,dict) and set(sources)==KERNEL_FILES
            if source_bound:
                for name,expected in sources.items():
                    if not name.startswith('src/survival_toolkit/') or '/' in name.removeprefix('src/survival_toolkit/'):
                        raise ValueError('Invalid qualification kernel path')
                    current=Path(__file__).parent/name.removeprefix('src/survival_toolkit/')
                    canonical=current.read_bytes().replace(b'\r\n',b'\n')
                    source_bound=source_bound and hashlib.sha256(canonical).hexdigest()==expected
            sources_match=source_bound
        evaluated=evaluated and source_bound
        return {'policy_version':registry['policy_version'],'status':profile['status'] if evaluated else 'not_evaluated',
                'requested_method_version':method_version,'evaluated_method_version':study.get('method_version'),
                'kernel_sources_match':sources_match,
                'clinical_basis':basis,'registry_sha256':hashlib.sha256(raw).hexdigest(),
                'study_git_revision':study['study_git_revision'],'summary_sha256':study['summary_sha256'],
                'supported_conditions':profile.get('supported_conditions',[]) if evaluated else [],
                'qualification_dimensions':study.get('qualification_dimensions') if evaluated else None,
                'claim_boundary':study.get('claim_boundary'),
                'extension_limitations':study.get('extension_limitations'),
                'failed_conditions':profile.get('failed_conditions',[]) if evaluated else [],
                'previous_method_evidence':None if evaluated else {
                    'method_version':study.get('method_version'),'status':profile['status'],
                    'failed_conditions':profile.get('failed_conditions',[])}}
    except (OSError,ValueError,KeyError,TypeError,AttributeError):
        return {'policy_version':'marker-qualification/1','status':'not_evaluated','clinical_basis':basis,
                'failed_conditions':[]}


def _withhold(inference,gate):
    inference=copy.deepcopy(inference)
    inference['engineering_qualification']=gate
    if gate['status']!='passed_supported_conditions_only':
        inference['diagnostic_status_before_qualification']=inference.get('status')
        if inference.get('status')!='not_assessed': inference['status']='withheld'
        inference['allowed']=False
        reason='engineering_qualification_'+gate['status']+': conditional marker calculations are exploratory only'
        if reason not in inference.setdefault('reasons',[]): inference['reasons'].append(reason)
        inference['interpretation']='The clinical-basis profile did not qualify for routine conditional marker inference. Raw calculations and estimable prediction models are exploratory; diagnostics do not certify assumptions.'
    return inference


def _withhold_signature_summary(result):
    signature=result.get('signature')
    if signature:
        result.setdefault('exploratory_signature',copy.deepcopy(signature))
        signature['optimism_corrected_c']=None
        signature['correction_note']='The subsample gap describes the diagnosis-governed procedure; it does not validate an exploratory marker signature fitted in the full cohort. Use the fixed external prediction metrics as exploratory estimates.'


def qualify_marker_result(result):
    """Apply to a product/API result before figures, tables, reporting and recipe export."""
    if result.get('primary_lens')!='added_value': return result
    gate=qualification(result.get('clinical_basis','linear'),result.get('method_version'))
    result['inference']=_withhold(result['inference'],gate)
    if gate['status']=='passed_supported_conditions_only':
        if not result['inference'].get('allowed'):
            _withhold_signature_summary(result)
        recipe=result.get('locked_recipe')
        if recipe:
            from survival_toolkit.marker_evaluation import recipe_hash
            recipe['inference']=copy.deepcopy(result['inference']);recipe['recipe_hash']=recipe_hash(recipe)
        return result
    for row in result['marker_table']:
        row.setdefault('exploratory',{'added_value':copy.deepcopy(row['added_value']),
                                     'tier':row['tier'],'pattern':row['pattern'],'exact':copy.deepcopy(row.get('exact'))})
        row.update(tier='inference withheld',pattern='Inference withheld',inference_status='withheld')
        for key in ('p_value','q_bh','p_fwer','q_perm'): row['added_value'][key]=None
        if row.get('exact') and row['exact'].get('adjusted'):
            row['exact']={**row['exact'],'adjusted':{**row['exact']['adjusted'],'wald_p':None,'lr_p':None}}
    result['tier_counts']={key:0 for key in result['tier_counts']}
    result['tier_counts']['inference withheld']=len(result['marker_table'])
    _withhold_signature_summary(result)
    recipe=result.get('locked_recipe')
    if recipe:
        from survival_toolkit.marker_evaluation import recipe_hash
        recipe['inference']=copy.deepcopy(result['inference']);recipe['recipe_hash']=recipe_hash(recipe)
    return result


def qualify_external_result(validation,recipe):
    if not recipe.get('clinical',{}).get('columns'): return validation
    inference=validation['inference'];gate=qualification(recipe['clinical'].get('basis','linear'),inference.get('method_version'))
    validation['inference']=_withhold(inference,gate)
    if gate['status']=='passed_supported_conditions_only': return validation
    for row in validation['markers']:
        row.setdefault('exploratory',{'replication_p_holm':row.get('replication_p_holm'),'replicated':row.get('replicated'),
                                     'marginal':copy.deepcopy(row.get('marginal')),'adjusted':copy.deepcopy(row.get('adjusted'))})
        row['replication_p_holm']=None;row['replicated']=False;row['inference_status']=validation['inference']['status']
        for lens in ('marginal','adjusted'):
            if row.get(lens): row[lens]={**row[lens],'wald_p':None}
    validation.setdefault('notes',[]).append('The engineering profile is exploratory only; fixed prediction metrics do not reinstate marker inference.')
    return validation
