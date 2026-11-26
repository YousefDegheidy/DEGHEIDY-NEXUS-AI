import json
import pytest
from scripts.release_gate import evaluate_run,gate,CASES


def write(path,news=9,other=9,blank=False):
    d={'records':[{'case':name,'question':'test','answer':'real answer',
                   'runtime_error':None,'human_score':(None if blank else news if name=='07_news' else other),
                   'critical_error':False} for name in sorted(CASES)]}
    path.write_text(json.dumps(d),encoding='utf-8')


def test_requires_human_scores(tmp_path):
    p=tmp_path/'a.json';write(p,blank=True)
    with pytest.raises(ValueError,match='human_score'):evaluate_run(p)


def test_cannot_pass_news_1_of_10(tmp_path):
    p=tmp_path/'a.json';write(p,news=1)
    with pytest.raises(ValueError,match='news grade'):evaluate_run(p)


def test_two_real_scored_runs_required(tmp_path):
    p=tmp_path/'a.json';write(p)
    with pytest.raises(ValueError,match='distinct'):gate(p,p)


def test_two_good_runs_pass(tmp_path):
    a=tmp_path/'a.json';b=tmp_path/'b.json'
    write(a,news=8,other=9);write(b,news=8,other=8.5)
    r=gate(a,b)
    assert r['approved'] and r['run_1']>=8 and r['run_2']>=8


def test_critical_failure_blocks_release(tmp_path):
    p=tmp_path/'a.json';write(p)
    d=json.loads(p.read_text());d['records'][0]['critical_error']=True
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='critical'):evaluate_run(p)
