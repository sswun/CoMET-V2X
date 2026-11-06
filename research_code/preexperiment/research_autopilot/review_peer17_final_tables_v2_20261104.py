"""Independent source-pointer, units, mean/SD and CSV serialization check.

Uses stdlib statistics/math, not generator arithmetic. This is a table review,
not an independent experimental matching or bootstrap implementation.
"""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from peer17_v1_20261104 import ROOT,BASE,atomic,ev,load,now
OUT=ROOT/'docs/paper_draft_v0_20261103/final_tables_v2'

def run():
    ledger=load(OUT/'numeric_ledger.json');cache={};hashes={};issues=[];csvs={}
    for c in ledger['cells']:
        values=[]
        for operand in c['operands']:
            ref=operand['source'];path=ROOT/ref['path']
            if str(path) not in cache:
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                cache[str(path)]=json.loads(path.read_text()) if operand['format']=='json' else list(csv.DictReader(path.open()))
            assert hashes[str(path)]==ref['sha256'],path
            v=cache[str(path)]
            for key in operand['pointer']:v=v[key]
            values.append(float(v))
        op=c['operation']
        if op=='identity':r=values[0]
        elif op=='mean':r=statistics.fmean(values)
        elif op=='sample_SD':r=statistics.stdev(values)
        elif op=='max':r=max(values)
        elif op=='difference':r=values[0]-values[1]
        else:raise ValueError(op)
        r*=c['factor']
        if not math.isclose(r,c['value'],rel_tol=1e-12,abs_tol=1e-10):issues.append(dict(table=c['table'],row=c['row'],metric=c['metric'],computed=r,recorded=c['value']))
        table=c['table']
        if table not in csvs:csvs[table]=list(csv.DictReader((OUT/(table+'.csv')).open()))
        matches=[x for x in csvs[table] if x['row']==c['row'] and x['metric']==c['metric']]
        if len(matches)!=1 or not math.isclose(float(matches[0]['value']),r,rel_tol=1e-12,abs_tol=1e-10):
            issues.append(dict(table=table,row=c['row'],metric=c['metric'],error='CSV mismatch or duplicate key',matches=len(matches)))
        if c['metric'].startswith('AP') or c['metric'] in ('ap50','ap70'):
            assert 0<=c['value']<=100,(c['table'],c['value'])
    audit=dict(created_at=now(),status='passed' if not issues else 'discrepancies_retained',
        cells=len(ledger['cells']),issues=issues,sources=len(cache),tables=len(csvs),
        scope=__doc__,reviewer=ev(__file__),manifest=ev(OUT/'manifest.json'))
    atomic(OUT/'independent_numeric_audit.json',audit)
    return audit

if __name__=='__main__':print(json.dumps(run(),indent=2))
