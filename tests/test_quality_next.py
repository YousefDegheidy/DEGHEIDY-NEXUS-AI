"""Regression cases based on reviewed user failures, independent fixture drafts.
No GPU, network access, or real model-quality score.
"""
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import patch
from threading import Event
from src.quality_next import spec_for, violations, structured_stream, repeat_answer
from src.dialogue import fresh_memory, commit_answer, build_messages
from src.news_quality import check_official_claim, allowed_url, analytical, strict_story_digest, official_target
from src.realtime import search_web, WebEvidence, plan_search
import app

PLAN='أنا طالب AI سنة رابعة. اعملي خطة شهرين لوظيفة AI Engineer مدفوعة. كل شهر 4 مهام عملية ونتيجة قابلة للقياس. رشح وظيفتين.'
RAG='صمم نظام RAG لشركة عندها 10000 ملف PDF بيتحدثوا يومياً. النظام لازم يفتكر آخر 5 رسائل ويعرض مصدر ورقم صفحة كل إجابة. اشرح Architecture في 6 خطوات، وفرق بين RAG وFine-tuning وConversation Memory، واذكر 3 طرق لتقليل Hallucinations.'
GOOD='## الشهر 1\n1. اختبر النموذج\n2. وثق البيانات\n3. سجل زمن الاستجابة\n4. ارفع الكود\nالنتيجة: تقرير قياس زمن الاستجابة.\n## الشهر 2\n1. اكتب API\n2. اختبر الأخطاء\n3. جهز Docker\n4. سجل اختبار تحميل\nالنتيجة: 5 اختبارات ناجحة كهدف.\n## الوظائف\n1. Junior AI Engineer\n2. Junior ML Engineer'
BAD='شهر أول: طور نظام رؤية حاسوب. شهر ثاني: طور chatbot.'
NEWS='ابحث عن أهم 3 أخبار سياسية دولية خلال آخر 12 ساعة بتوقيت القاهرة. اذكر تاريخ النشر وتاريخ الحدث والملخص والرابط المباشر للمقال الأصلي لكل خبر. استبعد مقالات الرأي والتحليلات والأخبار المحلية.'
CLAIM='فيه منشور بيقول إن Google أعلنت رسمياً إغلاق Google Colab نهائياً الأسبوع القادم. تحقق من المصادر الرسمية.'
NOW=datetime(2026,11,24,23,20,tzinfo=ZoneInfo('Africa/Cairo'))

def test_counted_plan_detects_incomplete():
    s=spec_for(PLAN);assert (s.n,s.tasks,s.roles)==(2,4,2)
    assert violations(PLAN,BAD) and not violations(PLAN,GOOD)
    assert 'month_1_task_count' in violations(PLAN,GOOD.replace('4. ارفع الكود\n',''))

def test_one_repair_then_only_final_answer():
    calls=[]
    def generator(m,b):
        calls.append(m[0]['content']);yield BAD if len(calls)==1 else GOOD
    out=list(structured_stream(PLAN,fresh_memory(),generator,build_messages,500))
    assert out==[GOOD] and len(calls)==2 and 'first draft failed' in calls[1]

def test_two_failed_attempts_flagged():
    out=list(structured_stream(PLAN,fresh_memory(),lambda *_:iter([BAD]),build_messages,500))
    assert len(out)==1 and 'لم يكتمل التحقق البنيوي' in out[0]

def test_cancellation_no_retry():
    cancel=Event();calls=[]
    def stream(*_):calls.append(1);cancel.set();yield BAD
    assert not list(structured_stream(PLAN,fresh_memory(),stream,build_messages,500,cancel)) and len(calls)==1

def test_repeat_same_session_approved_static():
    m=commit_answer(fresh_memory(),PLAN,GOOD)
    chat=[{'role':'user','content':PLAN},{'role':'assistant','content':GOOD}]
    assert repeat_answer(PLAN,chat,m)==GOOD
    assert repeat_answer(PLAN,chat,fresh_memory()) is None
    with patch('torch.cuda.is_available',return_value=False), patch.object(app,'model_stream',side_effect=AssertionError('repeat generated again')):
        assert list(app.turn(PLAN,chat,m))[-1][0][-1]['content']==GOOD

def test_repeat_incomplete_never_cached():
    m=commit_answer(fresh_memory(),PLAN,BAD);chat=[{'role':'user','content':PLAN},{'role':'assistant','content':BAD}]
    assert repeat_answer(PLAN,chat,m) is None

def test_repeat_live_news_never_cached():
    m=commit_answer(fresh_memory(),NEWS,'old news');chat=[{'role':'user','content':NEWS},{'role':'assistant','content':'old news'}]
    assert repeat_answer(NEWS,chat,m) is None

def test_repeat_not_after_changed_context():
    m=commit_answer(fresh_memory(),PLAN,GOOD);chat=[{'role':'user','content':'new context'},{'role':'assistant','content':GOOD}]
    assert repeat_answer(PLAN,chat,m) is None

def test_rag_missing_explicit_memory_and_page_detected():
    spec=spec_for(RAG);assert (spec.n,spec.memory,spec.mitigations,spec.pages)==(6,5,3,True)
    a='## Architecture\n'+'\n'.join(f'{i}. خطوة' for i in range(1,7))+'\n## الفرق\nRAG / Fine-tuning / Conversation Memory\n## تقليل الهلوسة\n1. دليل\n2. مصدر\n3. رفض'
    assert set(violations(RAG,a))=={'explicit_memory_limit','page_citation'}
    assert not violations(RAG,a.replace('1. خطوة','1. احفظ آخر 5 رسائل مع رقم الصفحة'))

def record(title,url,publisher):
    return {'url':url,'title':title,'publisher':publisher,'date_kind':'published','published':'2026-11-24T22:15:00+02:00'}

def test_analysis_question_title_excluded_before_reading():
    raw=[record('What Did Xi Jinping Get From the Shenzhen Summit?','https://foreignpolicy.com/x','Foreign Policy'),record('US and China announce diplomatic talks','https://publisher.example/a','News')]
    e=search_web(NEWS,search_fn=lambda _:raw,now=NOW)
    assert len(e.sources)==1 and 'announce' in e.sources[0]['title']

def test_analysis_path_after_decode():
    assert analytical({'title':'US and China leaders meet','url':'https://news.google.com/rss/x','article_url':'https://paper.example/analysis/summit','publisher':'Paper'})

def test_strict_news_excludes_unreadable_and_aggregated():
    s={'title':'US and China announce talks','url':'https://paper.example/a','publisher':'Paper','date_kind':'published','published':'2026-11-24 22:00 EET','read_status':'article_read','article_text':'Officials announced that representatives from both countries would attend talks next month to discuss commercial arrangements in a joint statement.'}
    e=WebEvidence('found','now','mock',[s,dict(s,title='Why the summit matters',url='https://paper.example/b'),dict(s,title='NATO ministers meet',url='https://news.google.com/rss/a'),dict(s,title='EU announces vote',url='https://paper.example/c',read_status='headline_only')],plan=plan_search(NEWS,now=NOW))
    out=strict_story_digest(NEWS,e);assert '1 من 3' in out and 'https://paper.example/a' in out and 'https://paper.example/c' not in out and 'news.google.com' not in out and 'Why the summit' not in out

def test_official_domain_boundaries():
    domains,_=official_target(CLAIM)
    assert allowed_url('https://research.google.com/colaboratory/faq.html',domains)
    assert not allowed_url('https://research.google.com.fake.example/x',domains)
    assert not allowed_url('https://evil.example@research.google.com/x',domains)
    assert not allowed_url('http://research.google.com/x',domains)

def test_official_reads_only_allowed_search_results():
    urls=[];queries=[]
    def search(q):queries.append(q);return [{'href':'https://research.google.com/colaboratory/faq.html','title':'Colab FAQ'},{'href':'https://research.google.com.fake.example/x','title':'spoof'}]
    def fetch(u,domains):urls.append(u);return 'The service is offered subject to resource constraints. Some runtimes can terminate automatically.'
    a=check_official_claim(CLAIM,search=search,fetch=fetch)
    assert len(queries)==4 and all('site:' in q for q in queries)
    assert urls==['https://research.google.com/colaboratory/faq.html'] and 'Colab FAQ' in a and 'spoof' not in a and 'غير مُثبت' in a

def test_official_no_access_is_not_falsehood():
    def error(q):raise OSError('network blocked')
    a=check_official_claim(CLAIM,search=error,fetch=lambda *_:'')
    assert '0/4' in a and 'غير مُثبت' in a and 'لا أستنتج نفيًا قاطعًا' in a

def test_edit_only_one_month_no_jobs():
    q='عندي مشروع Gym ومشروع Chatbot. عدّل الشهر الثاني فقط من خطتك السابقة إلى 3 مهام، وخلي المشروعين منفصلين. متعيدش الشهر الأول ومتقترحش وظائف جديدة.'
    spec=spec_for(q);assert (spec.kind,spec.n,spec.tasks)==('edit',2,3)
    a='## الشهر 2\n1. Gym\n2. Chatbot\n3. اختبارات لكل مشروع'
    assert not violations(q,a)
    assert violations(q,a+'\n## الوظائف\n1. Junior Engineer')
