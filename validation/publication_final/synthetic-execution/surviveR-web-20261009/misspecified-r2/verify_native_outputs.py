"""Verify the supplementary median-grouped web target against independent R.

Native Cox CSV values are rounded to four decimals. They do not support a
solver-precision claim. The original conditional residual diagnostic remains
a different target. Earlier unverified session attempts are retained.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'native-median-60'
REF = ROOT / 'independent-group-reference'
read = lambda p: list(csv.DictReader(p.open()))
rows = []
for item in read(ROOT / 'independent-R-conversion-checks.csv'):
    rows.append(dict(target='native median conversion', quantity=item['check'],
                     observed=item['observed'], reference=item['expected'],
                     criterion='prespecified conversion check',
                     solver_precision_comparable=False, passed=item['passed']=='TRUE'))
reference = {r['group']: r for r in read(REF / 'cox-60.csv')}
names = {'exp(coef)':'HR', 'exp(-coef)':'inverse_HR',
         'lower .95':'lower', 'upper .95':'upper', 'p.value':'p'}
web = read(RAW / 'SurviveR_COXHR.csv')
assert len(web) == 1 and web[0]['group'] == 'HIGH'
for column, ref_column in names.items():
    actual = float(web[0][column])
    expected = float(reference['HIGH'][ref_column])
    rows.append(dict(target='median-grouped Cox; reference LOW', quantity=column,
                     group='HIGH', observed=actual, reference=expected,
                     absolute_difference=abs(actual-expected),
                     criterion='four-decimal display rounding',
                     solver_precision_comparable=False, passed=actual==round(expected,4)))
text = '\n'.join(p.extract_text() for p in PdfReader(RAW / 'SurviveR_RiskCumEvTable.pdf').pages)
lines = re.findall(r'^(\d+(?:[ \t]+\d+){6})(?=[A-Z]*$)', text, flags=re.M)
assert len(lines) == 4, 'Unexpected native count-table layout; refuse extraction'
tokens = [[int(v) for v in line.split()] for line in lines]
km = read(REF / 'KM-60.csv')
# HIGH/LOW row order was checked against the unmodified rendered native PDF.
for quantity, field, offset in [('number at risk','risk',0), ('cumulative events','cumulative_events',2)]:
    for gi, group in enumerate(['HIGH','LOW']):
        for ti, t in enumerate([0,10,20,30,40,50,60]):
            ref = next(r for r in km if r['group']==group and float(r['time'])==t)
            actual = tokens[offset+gi][ti]
            expected = int(float(ref[field]))
            rows.append(dict(target='median-grouped KM', quantity=quantity, group=group,
                             time=t, observed=actual, reference=expected,
                             absolute_difference=abs(actual-expected),
                             criterion='exact integer equality',
                             solver_precision_comparable=False, passed=actual==expected))
logrank = read(RAW / 'SurviveR_LogRankTest.csv')
assert len(logrank)==1 and logrank[0]['']=='LOW'
actual = float(logrank[0]['HIGH'])
expected = float(read(REF / 'logrank-60.csv')[0]['raw_p'])
rows.append(dict(target='two-group log-rank', quantity='p', group='HIGH/LOW',
                 observed=actual, reference=expected, absolute_difference=abs(actual-expected),
                 criterion='maximum absolute difference 1e-6',
                 solver_precision_comparable=True, passed=abs(actual-expected)<=1e-6))
columns = list(dict.fromkeys(k for row in rows for k in row))
with (ROOT / 'native-output-checks.csv').open('w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
assert len(rows)==43
paths = [*RAW.iterdir(), *REF.iterdir(), ROOT/'native-converted-table.csv',
         ROOT/'independent-R-conversion-checks.csv', ROOT/'pre-execution-lock.json',
         ROOT/'grouping-lock.json', ROOT/'independent-group-reference.R', Path(__file__)]
result = dict(passed=all(r['passed'] for r in rows), checks=len(rows),
              conversion_checks=9, exact_risk_counts=14, exact_cumulative_event_counts=14,
              rounded_cox_quantities=5, full_precision_logrank_checks=1,
              original_n=600, original_events=345, endpoint=60, endpoint_events=343,
              group_counts={'LOW':300,'HIGH':300}, reference='LOW',
              scope='Supplementary native median-grouped KM/Cox target. Original conditional marker-residual diagnostic not executed.',
              task_code='D', rounding_is_not_solver_precision=True,
              raw_KM_survival_probabilities_unverified=True,
              deployed_build_hash_unverified=True,
              earlier_unverified_attempt_preserved='misspecified-r1',
              source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in paths if p.is_file()})
(ROOT / 'native-output-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))
raise SystemExit(0 if result['passed'] else 1)
