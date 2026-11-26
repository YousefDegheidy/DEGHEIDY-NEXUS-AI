"""Conservative quality gate for two human-reviewed *real* Qwen/T4 runs.

Usage: python -m scripts.release_gate real_run1.json real_run2.json
Each acceptance-suite record must be assigned human_score (0-10) and
critical_error (bool) AFTER reading the real model output. This script
never fabricates scores or promotes deterministic mocked replies as Qwen quality.
"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import re

CASES={f'{i:02d}_{name}' for i,name in enumerate(('career','edit','buses','code','memory','rag','news','rumor','voice','stop'),1)}


def evaluate_run(path):
    p=Path(path)
    data=json.loads(p.read_text(encoding='utf-8'))
    records=data.get('records',[])
    if len(records)!=10 or {x.get('case') for x in records}!=CASES:
        raise ValueError(f'{p}: expected exactly 10 distinct acceptance cases')
    score=0.0
    for rec in records:
        name=rec['case']
        if rec.get('runtime_error') or rec.get('critical_error'):
            raise ValueError(f'{p}: runtime/critical error in {name}')
        if not rec.get('answer'):
            raise ValueError(f'{p}: missing real answer for {name}')
        grade=rec.get('human_score')
        if isinstance(grade,bool) or not isinstance(grade,(int,float)) or not 0<=grade<=10:
            raise ValueError(f'{p}: human_score 0..10 required after human review for {name}')
        if name=='07_news' and grade<8:
            raise ValueError(f'{p}: news grade {grade}/10 must be >=8 (core weakness)')
        score+=grade
    return round(score/10,2)


def gate(first,second):
    if Path(first).resolve()==Path(second).resolve():
        raise ValueError('Two distinct test runs are mandatory')
    a=evaluate_run(first);b=evaluate_run(second)
    return {'run_1':a,'run_2':b,'approved':a>=8 and b>=8}




def strict_gate(first,second):
    """Release-grade gate; older gate() remains for legacy diagnostic reports."""
    a,b=(json.loads(Path(p).read_text(encoding='utf-8')) for p in (first,second))
    if not a.get('run_id') or not b.get('run_id') or a['run_id']==b['run_id']:
        raise ValueError('Two independent run IDs are required')
    for key in ('code_sha256','suite_sha256','base_model','model_revision','adapter_sha256','runtime_versions','gpu'):
        if not a.get(key) or a.get(key)!=b.get(key):raise ValueError('Different or missing tested runtime: '+key)
    if not re.fullmatch(r'[a-f0-9]{40}', a['model_revision']):raise ValueError('Pin the exact model revision')
    for d in (a,b):
        if d.get('run_type')!='REAL_T4_APP':raise ValueError('Not an actual CUDA app test')
        for rec in d['records']:
            if type(rec.get('critical_error')) is not bool:raise ValueError('Explicit human critical_error true/false required')
            if rec['case']=='09_voice' and not rec.get('asr_diagnostics'):raise ValueError('Real audio diagnostic required')
            if rec['case']=='10_stop' and not (rec.get('stop_accepted') and rec.get('new_answer_exact_437') and rec.get('stop_phase')=='cancelled'):
                raise ValueError('Actual Stop and post-Stop test must pass')
    result=gate(first,second)
    return result

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('Usage: python -m scripts.release_gate run1.json run2.json')
    try:
        result=strict_gate(sys.argv[1],sys.argv[2])
        print(json.dumps(result,ensure_ascii=False))
        raise SystemExit(0 if result['approved'] else 2)
    except (ValueError,KeyError,TypeError) as exc:
        print('NOT APPROVED:',exc,file=sys.stderr)
        raise SystemExit(2)

