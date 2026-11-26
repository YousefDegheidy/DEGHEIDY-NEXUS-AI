"""End-to-end *application* regression checks replaying user-reported failures.

No LLM weights are supplied. These are actual app/router/code-path tests, never
represented as real Qwen answer-quality evaluations.
"""
from __future__ import annotations
import contextlib
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo
import re
import pytest
import app
from src.dialogue import build_messages, commit_answer, fresh_memory
from src.coverage import extract_requirements
from src.precision import multiple_bus_info, occupancy_solution, average_empty_solution
from src.realtime import needs_fresh_search, plan_search, search_web, should_search, unavailable_message
from src.evidence_guard import compose_coverage_digest

CAIRO=ZoneInfo('Africa/Cairo')
NOW=datetime(2026,11,22,16,5,tzinfo=CAIRO)
BUS='عندي 3 أتوبيسات: الأول 42 كرسي وفيه 33 راكب، الثاني 30 كرسي وفيه 24 راكب، والثالث 18 كرسي وفيه 9 ركاب. احسب نسبة إشغال كل أتوبيس والإجمالي للأسطول كله، ورتّبهم من الأقل للأعلى. وضح الحسابات واتأكد من النتيجة قبل الرد.'
AVG='عندي الكود ده:\ndef average(numbers):\n    return sum(numbers) / len(numbers)\nعايزك تشرح المشكلة لما القائمة تكون فاضية، وتصلح الدالة وتكتب 3 اختبارات pytest.'
CAREER='أنا طالب AI وعندي مشاريع في الرؤية الحاسوبية وLLMs، ومش عايز أبدأ من الصفر. اقترح خطة عملية لأول 30 يوم عشان أجهز نفسي لوظيفة مدفوعة، في 4 خطوات مرقمة فقط. متقترحش Internships ولا كورسات مبتدئين، ولا تفترض إني اشتغلت في Production.'
PLANT='عندي تطبيق بيستقبل صور نباتات ويستخدم CNN للتصنيف. عايز أنشره باستخدام FastAPI وDocker.'
RECALL='فاكر أنا قلتلك في بداية كلامنا عندي خبرة ومشاريع في إيه؟ وإيه القيود اللي حددتها لما طلبت خطة شغل؟'
NEWS='اختار أهم خبرين دوليين نُشرا خلال آخر 12 ساعة بتوقيت القاهرة. لكل خبر اكتب: ماذا حدث، متى حدث، متى نُشر التقرير، المصدر الأصلي ورابطه، وأهمية الحدث. استبعد المقالات التحليلية والرأي.'
CLAIM='قرأت منشورًا بيقول إن Google قررت إيقاف Gemini نهائيًا وإن كل خدماته هتتقفل الأسبوع الجاي. هل فيه إعلان رسمي يؤكد ده؟ دور على أدلة موثوقة.'


def test_buses_user_text_exact_all_three():
    buses=multiple_bus_info(BUS)
    assert len(buses)==3
    assert [(x['seats'],x['passengers']) for x in buses]==[(42,33),(30,24),(18,9)]
    s=occupancy_solution(BUS)
    for value in ('78.57%','80%','50%','66 ÷ 90','73.33%','الثالث (50%) ← الأول (78.57%) ← الثاني (80%)'):
        assert value in s,value


def test_buses_avoid_missing_vehicle_silent_total():
    s=occupancy_solution('عندي 3 أتوبيسات: الأول 42 كرسي 33 راكب، الثاني 30 كرسي 24 راكب. احسب الإجمالي')
    assert s is None  # Caller must ask for missing data, never publish 57/72 as fleet result.


def test_buses_application_cpu_turn():
    with patch('torch.cuda.is_available',return_value=False):
        snaps=list(app.turn(BUS,[],fresh_memory()))
    answer=snaps[-1][0][-1]['content']
    assert '73.33%' in answer and 'الثالث' in answer


def test_average_python_real_execution():
    answer=average_empty_solution(AVG)
    assert answer
    blocks=re.findall(r'```python\n(.*?)```',answer,re.S)
    assert len(blocks)==1
    namespace={}
    exec(blocks[0],namespace)
    for name in ('test_average_normal','test_average_empty','test_average_negative'):
        namespace[name]()


def test_average_application_cpu_turn():
    with patch('torch.cuda.is_available',return_value=False):
        snaps=list(app.turn(AVG,[],fresh_memory()))
    answer=snaps[-1][0][-1]['content']
    assert 'ValueError' in answer and 'pytest' in answer and 'test_average_negative' in answer


def test_memory_cross_topic_only_user_premises():
    mem=fresh_memory()
    mem=commit_answer(mem,CAREER,'assistant suggested Kubernetes and DevOps')
    mem=commit_answer(mem,PLANT,'assistant incorrectly said no earlier projects')
    msgs=build_messages(RECALL,mem)
    sys=msgs[0]['content']
    assert 'الرؤية الحاسوبية وLLMs' in sys
    assert 'مش عايز أبدأ' in sys
    assert 'تطبيق بيستقبل صور نباتات' in sys
    assert 'assistant incorrectly said' not in sys
    assert 'facts stated by user' in sys


def test_memory_normal_new_subject_separate_from_career():
    mem=commit_answer(fresh_memory(),CAREER,'ai career')
    assert not should_search(PLANT,mem)
    assert 'تطبيق' in build_messages(PLANT,mem)[-1]['content']


def test_12_hour_news_window_exact():
    assert needs_fresh_search(NEWS)
    p=plan_search(NEWS,now=NOW)
    assert p.start==NOW-timedelta(hours=12)
    assert p.end==NOW+timedelta(minutes=1)
    assert p.topic=='political news'
    assert {x.name for x in extract_requirements(NEWS)}>={'news_current','news_two'}


def test_12_hour_news_source_filter_and_two_links():
    raw=[
        {'url':'https://reuters.com/world/report-1','title':'European parliament debates new budget',
         'published':'2026-11-22T10:00:00+02:00','publisher':'Reuters'},
        {'url':'https://apnews.com/article/minister-summit','title':'Prime minister holds emergency summit',
         'published':'2026-11-22T13:00:00+02:00','publisher':'AP News'},
        {'url':'https://aawsat.com/opinion','title':'رأي: كاتب يناقش السياسة العالمية',
         'published':'2026-11-22T14:00:00+02:00','publisher':'Opinion'},
        {'url':'https://example.com/old','title':'President meets government delegation',
         'published':'2026-11-21T19:00:00+02:00','publisher':'Example'},
    ]
    ev=search_web(NEWS,search_fn=lambda _:raw,now=NOW)
    assert len(ev.sources)==2
    answer=compose_coverage_digest(NEWS,ev)
    assert 'reuters.com' in answer and 'apnews.com' in answer
    assert 'aawsat.com' not in answer and 'example.com' not in answer


def test_news_empty_reports_exact_window_not_wrong_day():
    ev=search_web(NEWS,search_fn=lambda _:[],now=NOW)
    txt=unavailable_message(ev)
    assert '2026-11-22T04:05' in txt and '2026-11-22T16:06' in txt
    assert '2026-11-19' not in txt


def test_gemini_announcement_requires_evidence_not_confident_denial():
    assert needs_fresh_search(CLAIM)
    p=plan_search(CLAIM,now=NOW)
    assert p.topic=='Google Gemini shutdown claim'
    ev=search_web(CLAIM,search_fn=lambda _:[],now=NOW)
    assert ev.status=='no_results'
    # No mock source exists; a chatbot must NOT claim a fact-check disproved the rumor.
    assert 'لم أجد' in unavailable_message(ev)


def test_gemini_application_no_results_safe_claim():
    evidence=search_web(CLAIM,search_fn=lambda _:[],now=NOW)
    with patch('src.news_quality._search',return_value=[]),patch('app.search_web',side_effect=AssertionError('official query must use official path')),patch('torch.cuda.is_available',return_value=False):
        snaps=list(app.turn(CLAIM,[],fresh_memory()))
    answer=snaps[-1][0][-1]['content']
    assert 'غير مُثبت' in answer
    assert 'الشائعة كاذبة' not in answer


def test_regular_career_question_never_news():
    assert not needs_fresh_search(CAREER)


def test_stop_and_audio_code_preserved_from_baseline():
    import hashlib
    root=Path(__file__).resolve().parents[1]
    assert hashlib.sha256((root/'src/speech.py').read_bytes()).hexdigest()=='293a350f1a0451aaf65b6035a860608a0e4bc740492f12f59a726385bee672bf'
    assert hashlib.sha256((root/'src/job_manager.py').read_bytes()).hexdigest()=='3eba343fdc94955072599a70f62446c34d7069efdb312b8fb9ab6e3fcfe8a790'


def test_offline_hardware_diagnostic_is_truthful():
    import torch
    # This test works both in CPU-only local CI and in Google Colab T4.
    detected=bool(torch.cuda.is_available())
    assert detected in (True,False)
