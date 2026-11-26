import json,re
from pathlib import Path
from unittest.mock import patch
import pytest
from src.precision import occupancy_solution,mutable_default_solution,exact_short_calculation,stable_unique_solution
from src.evidence_guard import evaluate_news_answer,compose_coverage_digest
from src.dialogue import fresh_memory
from src.realtime import WebEvidence,plan_search
from tests.test_evidence_guard import ev_obj
from tests.test_targeted_74_news_regressions import PROMPT,NOW,record
from src.realtime import search_web,evidence_footer
from scripts.release_gate import strict_gate,CASES
from training.train_t4_lora import audit_training_data,validate_plan,encode_conversation,collate_assistant_only
from src.qwen_adapter import validate_adapter_base

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('q,expected',[('احسب 19 × 23 واكتب الرقم فقط','437'),('احسب ٢٩ × ١٨ واكتب الناتج فقط»','522'),('calculate -4 * 9, return the result only','-36')])
def test_exact_arithmetic(q,expected):assert exact_short_calculation(q)==expected

def test_arithmetic_does_not_discard_other_constraints():assert exact_short_calculation('احسب 19 × 23 واشرح الطريقة') is None

def test_four_buses_colloquial():
    answer=occupancy_solution('عندي 4 أتوبيسات: الأول 32 كرسي وراكب فيه 26، والثاني 40 كرسي وفيه 34 راكب، والتالت 25 كرسي و17 راكب، والرابع 25 كرسي وفيه 20 راكب. احسب إشغال كل أتوبيس، وإجمالي إشغال الأسطول، ورتب الأتوبيسات من الأقل للأعلى.')
    assert answer and all(x in answer for x in ('81.25%','85%','68%','80%','79.51%','97 ÷ 122'))

def _check_generated_code(answer):
    # Execute ONLY trusted, locally generated test fixtures; never arbitrary user code.
    parts=re.findall(r'```python\n(.*?)\n```',answer,re.S)
    assert len(parts)==2
    ns={};exec('\n'.join(parts),ns)
    tests=[x for x in ns if x.startswith('test_')]
    assert len(tests)==3
    for name in tests:ns[name]()

def test_generic_mutable_default():
    q='راجع الكود: def save_result(value, results=[]):\n results.append(value)\n return results\n اشرح واكتب pytest لقيمة سالبة'
    a=mutable_default_solution(q);assert 'results=None' in a;assert '(-5)' in a;_check_generated_code(a)

def test_different_code_not_silently_rewritten():assert mutable_default_solution('راجع def add_item(x, items=[]): return len(items)') is None

def test_stable_unique():_check_generated_code(stable_unique_solution('راجع def unique_items(items):\n return list(set(items))\n مع الحفاظ على ترتيب أول ظهور'))

def test_hallucinated_link_blocks_draft():
    a=evaluate_news_answer('Trump arrived in China on Wednesday [1]. https://fabricated.example/a',ev_obj());assert not a.approved

def test_link_cannot_bypass_claim_check():
    a=evaluate_news_answer('Trump arrived in China on Wednesday [1]. Trump signed an agreement on Wednesday [1] https://example.org/trump-china',ev_obj());assert not a.approved

def test_index_link_not_printed_if_original_required():
    ev=search_web(PROMPT,search_fn=lambda _: [record('https://news.google.com/rss/articles/abc','European leaders announce summit')],now=NOW)
    assert 'https://news.google.com' not in evidence_footer(ev)
    assert 'ليس رابط الناشر' in evidence_footer(ev)

def test_filtered_citation_id_matches_original_evidence():
    ev=ev_obj();ev.sources.insert(0,{'title':'Earlier item','url':'https://example.org/a','publisher':'Other','published':'2026-11-18','date_kind':'published','read_status':'headline_only'})
    out=compose_coverage_digest('هات أهم خبرين سياسيين وملخص من سطرين وتاريخ وقوع الحدث',ev)
    assert '[المقال لدى الناشر](https://example.org/trump-china) [2]' in out
    assert 'تاريخ وقوع الأحداث' in out

def test_failed_fresh_search_clears_old_news_state():
    import app
    ev=ev_obj();failed=WebEvidence('no_results','now','fixture',[],plan=plan_search(PROMPT,now=NOW))
    with patch('app.should_search',return_value=True),patch('app.search_web',return_value=failed):
        last=list(app.turn(PROMPT,[],fresh_memory(),ev))[-1]
    assert last[3] is None

def test_synthetic_dataset_audit():
    rows=audit_training_data(ROOT/'training/data/qa74_supervised.jsonl');assert len(rows)==140

def test_t4_plan():
    validate_plan('Qwen/Qwen3.5-4B',14.7,768,8,1,8)
    with pytest.raises(ValueError):validate_plan('Qwen/Qwen3.5-4B',8,768,8,1,8)

class MockTokenizer:
    """Character-offset fixture; no actual HF tokenizer or GPU in this unit test."""
    def apply_chat_template(self,messages,**kwargs):
        return ''.join('<|im_start|>'+m['role']+'\n'+('<think>\n\n</think>\n\n' if m['role']=='assistant' and i==len(messages)-1 else '')+m['content'].strip()+'<|im_end|>\n' for i,m in enumerate(messages))
    def __call__(self,text,**kwargs):return {'input_ids':[ord(x) for x in text],'offset_mapping':[(i,i+1) for i in range(len(text))]}
    def decode(self,ids):return ''.join(map(chr,ids))

MASK_MESSAGES=[{'role':'system','content':'RULE'},{'role':'user','content':'QUESTION'},{'role':'assistant','content':'ANSWER'}]

def test_only_assistant_labels_and_padding():
    tok=MockTokenizer();x=encode_conversation(tok,MASK_MESSAGES,256)
    assert tok.decode([i for i in x['labels'] if i!=-100])=='ANSWER<|im_end|>'
    y={'input_ids':[5],'attention_mask':[1],'labels':[5]};batch=collate_assistant_only([x,y],0)
    assert batch['labels'][1].tolist()==[5]+[-100]*(len(x['labels'])-1)

def test_long_training_example_rejected():
    with pytest.raises(ValueError):encode_conversation(MockTokenizer(),MASK_MESSAGES,3)

def test_adapter_revision_match(tmp_path):
    (tmp_path/'adapter_config.json').write_text(json.dumps({'base_model_name_or_path':'Qwen/Qwen3.5-4B'}));(tmp_path/'adapter_model.safetensors').write_bytes(b'fixture')
    (tmp_path/'training_manifest.json').write_text(json.dumps({'base_model':'Qwen/Qwen3.5-4B','base_model_revision':'a'*40}))
    with pytest.raises(ValueError):validate_adapter_base(tmp_path,'Qwen/Qwen3.5-4B','b'*40)
    validate_adapter_base(tmp_path,'Qwen/Qwen3.5-4B','a'*40)

def gate_fixture(id):
    return {'run_type':'REAL_T4_APP','run_id':id,'code_sha256':'fixture','suite_sha256':'fixture','base_model':'Qwen/Qwen3.5-4B','model_revision':'a'*40,'adapter_sha256':'BASE_NO_ADAPTER','runtime_versions':{'torch':'fixture'},'gpu':'T4','records':[{'case':name,'answer':'fixture only','human_score':9,'critical_error':False,'runtime_error':None,'asr_diagnostics':'fixture','stop_accepted':True,'stop_phase':'cancelled','new_answer_exact_437':True} for name in sorted(CASES)]}

def test_strict_gate_rejects_duplicate_run_and_config(tmp_path):
    a,b=tmp_path/'a.json',tmp_path/'b.json';d=gate_fixture('one');a.write_text(json.dumps(d));b.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='independent'):strict_gate(a,b)
    d['run_id']='two';d['adapter_sha256']='DIFFERENT';b.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='runtime'):strict_gate(a,b)

def test_gate_does_not_accept_missing_voice_or_stop(tmp_path):
    a,b=tmp_path/'a.json',tmp_path/'b.json';a.write_text(json.dumps(gate_fixture('one')));d=gate_fixture('two');d['records'][8]['asr_diagnostics']='';b.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='audio'):strict_gate(a,b)
    d=gate_fixture('two');d['records'][9]['stop_phase']='finished';b.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='Stop'):strict_gate(a,b)

def test_gate_accepts_complete_fixture_only(tmp_path):
    a,b=tmp_path/'a.json',tmp_path/'b.json';a.write_text(json.dumps(gate_fixture('one')));b.write_text(json.dumps(gate_fixture('two')))
    assert strict_gate(a,b)['approved']  # unit-test fixture only; not actual model certification

def test_model_draft_cannot_offer_index_as_direct():
    from src.evidence_guard import render_news_response
    ev=ev_obj();ev.sources[0]['url']='https://news.google.com/rss/a'
    text,verdict=render_news_response('هات خبر ورابط مباشر للمصدر الأصلي','Trump arrived in China on Wednesday [1]. https://news.google.com/rss/a',ev)
    assert not verdict.approved and 'https://news.google.com' not in text
