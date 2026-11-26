"""Offline regressions and deterministic stress cases. Not model scores."""
from decimal import Decimal
from random import Random
from unittest.mock import patch
from src.precision import occupancy_solution, transport_additions
from src.news_quality import official_target, check_official_claim, analytical, strict_story_digest
from src.realtime import WebEvidence
from src.quality_next import violations, spec_for
from src.dialogue import fresh_memory, commit_answer, build_messages
import app

BUS='عندي 4 أتوبيسات: الأول 40 كرسي و30 راكب، الثاني 50 كرسي و35 راكب، الثالث 30 كرسي و27 راكب، الرابع 20 كرسي و12 راكب. احسب إشغال كل أتوبيس وإجمالي إشغال الأسطول ورتبهم تصاعدياً.'
CLAIM='NVIDIA أعلنت رسمياً إيقاف دعم CUDA لجميع كروت RTX. تحقق من موقع NVIDIA الرسمي ووثائق CUDA الرسمية.'

def test_compound_arithmetic_stress_1000():
    rng=Random(811)
    for i in range(1000):
        cost=rng.randint(1,999)*1000; prod=rng.randint(1,999)*1000; rate=rng.randint(0,100)
        q=BUS+f' تكلفة النقل {cost} جنيه والإنتاج {prod} متر مربع. احسب تكلفة النقل لكل متر مربع والتوفير لو خفضنا التكلفة بنسبة {rate}%.'
        a=occupancy_solution(q)
        expected=format((Decimal(cost)/Decimal(prod)).quantize(Decimal('0.0001')).normalize(),'f')
        assert '74.29%' in a and f'**{expected} جنيه/م²**' in a and f'**{cost*rate//100} جنيه**' in a, i

def test_ambiguous_and_zero_production():
    assert 'غير معرّفة' in ' '.join(transport_additions('احسب تكلفة النقل لكل متر مربع. 1200 جنيه والإنتاج 0 متر مربع.'))
    assert 'محتاج' in ' '.join(transport_additions('احسب التوفير لو خفضنا التكلفة 15%.'))

def test_official_cuda_queries_filter_irrelevant():
    seen=[]
    def search(q):
        seen.append(q)
        return [{'href':'https://docs.nvidia.com/cuda/release-notes/index.html','title':'CUDA release notes'}, {'href':'https://nvidia.com/company/careers','title':'Company jobs'}, {'href':'https://docs.nvidia.com.evil.test/cuda','title':'CUDA fake'}]
    a=check_official_claim(CLAIM,search=search,fetch=lambda *_:'CUDA release notes describe supported toolkits and hardware.')
    assert all('CUDA' in q for q in seen) and any('RTX' in q for q in seen)
    assert 'CUDA release notes' in a and 'Company jobs' not in a and 'evil.test' not in a and 'غير مُثبت' in a

def test_official_route_independent_of_generic_router():
    with patch.object(app,'should_search',return_value=False),patch.object(app,'check_official_claim',return_value='official evidence') as check:
        out=list(app.turn(CLAIM,[],fresh_memory()))
    check.assert_called_once_with(CLAIM)
    assert out[-1][0][-1]['content']=='official evidence'

def test_analysis_titles_and_metadata():
    assert analytical({'title':'The limits of global climate diplomacy'})
    assert analytical({'title':'Diplomats meet','genre':'opinion'})
    assert not analytical({'title':'Diplomats meet to discuss treaty','url':'https://example.org/news/treaty'})

def test_two_sentence_requirement_fail_closed():
    q='3 أخبار سياسية استبعد مقالات الرأي والتحليلات. ملخص من جملتين ورابط المقال الأصلي.'
    s={'title':'Officials meet for talks','url':'https://example.org/news/talks','publisher':'Paper','date_kind':'published','published':'today','read_status':'article_read','article_text':'Officials from both countries met to discuss plans for a trade agreement.'}
    e=WebEvidence('found','now','mock',[s])
    assert '0 من 3' in strict_story_digest(q,e)
    s['article_text']+=' Both delegations agreed to schedule another meeting in the coming months.'
    a=strict_story_digest(q,e)
    assert '1 من 3' in a and 'Both delegations' in a and 'https://example.org/news/talks' in a

def test_specific_rag_checks():
    q='صمم RAG Architecture في 2 خطوات تشمل Incremental Indexing وحذف الملفات القديمة وصلاحيات الوصول وآخر 5 رسائل لكل مستخدم. اذكر 2 طرق لتقليل Hallucinations.'
    a='## Architecture\n1. جمع البيانات\n2. آخر 5 رسائل لكل مستخدم\n## الفرق\nRAG Fine-tuning Conversation Memory\n## تقليل الهلوسة\n1. تحقق\n2. دليل'
    assert {'delete_obsolete_versions','incremental_indexing','access_before_retrieval'}<=set(violations(q,a))
    assert not violations(q,a.replace('جمع البيانات','Incremental Indexing وحذف الملفات القديمة وتطبيق صلاحيات الوصول قبل الاسترجاع'))

def test_numeric_plan_requires_number():
    q='خطة شهرين لوظيفة AI Engineer كل شهر 2 مهام ونتيجة رقمية قابلة للقياس.'
    a='## الشهر 1\n1. مهمة\n2. مهمة\nالنتيجة: مشروع\n## الشهر 2\n1. مهمة\n2. مهمة\nالنتيجة: 5 اختبارات'
    assert 'month_1_numeric_target' in violations(q,a)

def test_recall_prompt_only_user_evidence():
    m=commit_answer(fresh_memory(),'أنا طالب AI سنة رابعة وعندي Python','TensorFlow expert')
    out=build_messages('لخص المهارات اللي أنا قلتها بنفسي من بداية المحادثة',m)
    assert 'سنة رابعة' in out[0]['content'] and not any(x['role']=='assistant' for x in out)

def test_partial_edit_retains_two_valid_tasks():
    from src.quality_next import structured_stream
    q='عدّل الشهر الثاني فقط إلى 3 مهام بالضبط.'
    draft='## الشهر 2\n1. Dockerize project A\n2. Test project B'
    calls=[]
    def stream(msgs,budget):
        calls.append((msgs,budget))
        yield draft if len(calls)==1 else '3. Document both projects independently'
    out=list(structured_stream(q,fresh_memory(),stream,build_messages,500))
    assert out[-1]==draft+'\n3. Document both projects independently' and len(calls)==2 and calls[-1][1]<=360
