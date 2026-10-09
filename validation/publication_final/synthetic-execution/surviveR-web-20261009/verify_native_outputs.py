"""Check preserved web downloads against independently calculated R values.

The web HR CSV is rounded to four decimals. Matching its rounding is not a
solver comparison at 1e-6. Risk counts are exact integers; log-rank output
is compared to a BH-adjusted R reconstruction, labelled as such.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent
RAW=ROOT/'native-grade-60'
REF=ROOT/'surviveR-independent-native-reference-r1'
read=lambda p:list(csv.DictReader(p.open()))
rows=[]
web=read(RAW/'SurviveR_COXHR.csv')
reference={r['group']:r for r in read(REF/'cox-60.csv')}
names={'exp(coef)':'HR','exp(-coef)':'inverse_HR','lower .95':'lower','upper .95':'upper','p.value':'p'}
for row in web:
    for column,ref_column in names.items():
        actual=float(row[column]); expected=float(reference[row['group']][ref_column])
        rows.append(dict(target='grade-only Cox; reference A',quantity=column,group=row['group'],
            observed=actual,reference=expected,absolute_difference=abs(actual-expected),
            criterion='four-decimal display rounding',solver_precision_comparable=False,
            passed=actual==round(expected,4)))
text='\n'.join(p.extract_text() for p in PdfReader(RAW/'SurviveR_RiskCumEvTable.pdf').pages)
numeric_lines=re.findall(r'^(\d+(?:[ \t]+\d+){6})(?=[ABC]?$)',text,flags=re.M)
assert len(numeric_lines)==6, 'Expected exactly six fixed-width count rows'
tokens=[int(v) for line in numeric_lines for v in line.split()]
assert len(tokens)==42, 'Unexpected risk/cumulative-event PDF layout; refuse extraction'
km=read(REF/'KM-60.csv')
for gi,group in enumerate(['A','B','C']):
    for ti,t in enumerate([0,10,20,30,40,50,60]):
        ref=next(r for r in km if r['group']==group and float(r['time'])==t)
        actual=tokens[7*gi+ti]; expected=int(ref['risk'])
        rows.append(dict(target='grade-stratified KM',quantity='number at risk',group=group,time=t,
            observed=actual,reference=expected,absolute_difference=abs(actual-expected),
            criterion='exact integer equality',solver_precision_comparable=False,passed=actual==expected))
web_log=read(RAW/'SurviveR_LogRankTest.csv')
for ref in read(REF/'logrank-60.csv'):
    actual=float(next(r for r in web_log if r['']==ref['second'])[ref['first']])
    expected=float(ref['BH_p']);diff=abs(actual-expected)
    rows.append(dict(target='pairwise log-rank BH reconstruction',quantity='p',group=ref['first']+'/'+ref['second'],
        observed=actual,reference=expected,absolute_difference=diff,criterion='maximum absolute difference 1e-6',
        solver_precision_comparable=True,passed=diff<=1e-6))
columns=list(dict.fromkeys(k for row in rows for k in row))
with (ROOT/'native-output-checks.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [*RAW.iterdir(),*REF.iterdir(),ROOT/'pre-execution-lock.json',ROOT/'native-reference-scope.json',ROOT/'independent-native-reference.R'] if p.is_file()}
result=dict(passed=all(r['passed'] for r in rows),checks=len(rows),exact_risk_counts=21,
    rounded_cox_quantities=10,full_precision_logrank_reconstruction=3,n=260,events=151,
    scope='Implementation reference for the native unadjusted grade workflow; original adjusted HR target and raw KM survival estimates remain unverified',
    rounding_is_not_solver_precision=True,logrank_adjustment_is_reconstructed_not_build_verified=True,
    source_hashes=hashes)
(ROOT/'native-output-verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result));raise SystemExit(0 if result['passed'] else 1)
