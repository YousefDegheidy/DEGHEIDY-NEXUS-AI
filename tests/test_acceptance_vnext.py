"""Independent regression checks from the user's REAL failed ten-question transcript.

Pure Python tests: these are NOT a Qwen accuracy score.
"""
import hashlib
import re
from datetime import datetime,timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import app
from src.precision import multiple_bus_info,occupancy_solution
from src.coverage import extract_requirements,missing_requirements
from src.dialogue import build_messages,commit_answer,fresh_memory,choose_budget
from src.realtime import plan_search,search_web

CAREER="I'm a fourth-year AI student وعندي مشاريع في TensorFlow وComputer Vision وLLMs. عايز ألاقي شغل AI Engineer مدفوع، ومش عايز Internships ولا كورسات مبتدئين. اعملي خطة 3 شهور، لكل شهر 3 مهام عملية ونتيجة قابلة للقياس. وبعدها حدد أول وظيفتين مناسبين لمستوايا، من غير ما تفترض خبرة Production عندي."
CAREER_BAD='''شهر 1: قم بإنشاء مشروع... والنتيجة رابط Github
شهر 2: طور النموذج... والنتيجة تقرير
شهر 3: قدم مشروع... والنتيجة CV
الوظائف المناسبة: Junior Computer Vision Engineer، MLOps Trainee.'''
BUS='عندي 3 أتوبيسات: الأول سعته 36 كرسي وراكب فيه 27، والثاني سعته 30 كرسي وفيه 24 راكب، والثالث سعته 20 كرسي وفيه 11 راكب. احسب إشغال الأول والثاني والثالث، والإجمالي للأسطول، ورتبهم من الأقل إشغالًا للأعلى. راجع كل الأرقام قبل ما تجاوب.'
RAG='عندي Chatbot بيقرأ ملفات PDF بتتغير يوميًا، ولازم يفتكر آخر 5 رسائل. اشرح الفرق بين RAG وFine-tuning وConversation Memory بالمصري، وصمم Architecture من 5 خطوات تبدأ بتحديث الملفات وتنتهي برد مدعوم بمصدر. بعد كده اذكر 3 أخطاء تقنية لازم أتجنبها.'
NEW='ابحث عن أهم خبرين سياسيين دوليين حصل فيهما تطور خلال آخر 12 ساعة بتوقيت القاهرة. لكل خبر اذكر الحدث وتاريخ النشر وملخصًا واقعيًا ورابطًا مباشرًا. استبعد مقالات رأي.'


def test_actual_last_bus_question_all_three_and_total():
    assert len(multiple_bus_info(BUS))==3
    txt=occupancy_solution(BUS)
    for expected in ('الأول: 27 ÷ 36','الثاني: 24 ÷ 30','الثالث: 11 ÷ 20',
                     '75%','80%','55%','62 ÷ 86','72.09%',
                     'الثالث (55%) ← الأول (75%) ← الثاني (80%)'):
        assert expected in txt,expected


def test_actual_last_bus_question_cpu_app_does_not_refuse():
    with patch('torch.cuda.is_available',return_value=False):
        frames=list(app.turn(BUS,[],fresh_memory()))
    answer=frames[-1][0][-1]['content']
    assert '72.09%' in answer and 'بيانات الأتوبيسات مش كاملة' not in answer


def test_incomplete_second_bus_fails_closed():
    q='3 أتوبيسات: الأول 36 كرسي و27 راكب، الثاني 30 كرسي، والثالث 20 كرسي 11 راكب. احسب الإجمالي'
    assert multiple_bus_info(q) is None


def test_repeated_ordinal_in_instruction_not_a_bus():
    q='3 أتوبيسات: الأول 40 كرسي و30 راكب، الثاني 30 كرسي و20 راكب، والثالث 20 كرسي و10 ركاب. رتب الأول والثاني والثالث'
    assert len(multiple_bus_info(q))==3
    assert '60 ÷ 90' in occupancy_solution(q)


def test_three_months_no_hidden_six_month_instruction():
    sys=build_messages(CAREER,fresh_memory())[0]['content']
    assert 'career' not in sys.lower() or 'month' in sys.lower()
    assert 'Provide months 1 through 6' not in sys
    assert choose_budget(CAREER)>=1200
    assert '3 tasks' in sys or '9 tasks total' in sys


def test_bad_previous_answer_detected_as_structurally_incomplete():
    missing=missing_requirements(CAREER,CAREER_BAD)
    assert {'month_1_tasks_incomplete','month_2_tasks_incomplete','month_3_tasks_incomplete','two_roles_missing','prohibited_role_type'} <= set(missing)


def test_good_three_month_answer_passes_structure():
    good='''**الشهر الأول**
1. المهمة أ
2. المهمة ب
3. المهمة ج
**النتيجة:** رابط API قابل للفحص.
**الشهر الثاني**
1. مهمة د
2. مهمة هـ
3. مهمة و
**النتيجة:** تقرير قياس.
**الشهر الثالث**
1. مهمة ز
2. مهمة ح
3. مهمة ط
**النتيجة:** CV وطلبات توظيف.
**وظيفتان مدفوعتان:**
1. Junior AI Engineer
2. Junior Computer Vision Engineer.'''
    assert not missing_requirements(CAREER,good), missing_requirements(CAREER,good)


def test_one_month_edit_detects_other_month_drift():
    q='أنا كمان عندي Gym Exercise Analysis باستخدام MediaPipe وOpenCV ومشروع Multilingual Chatbot. عدّل الشهر الثاني فقط من الخطة اللي فاتت إلى 3 مهام مناسبة لمشاريعي. متعيدش باقي الخطة.'
    names={r.name for r in extract_requirements(q)}
    assert 'single_month_revision' in names and 'single_month_three_tasks' in names
    assert 'unrequested_months' in missing_requirements(q,'الشهر الثاني: 1. شيء\n2. شيء\n3. شيء\nالشهر الثالث: شيء')
    assert 'unrequested_months' not in missing_requirements(q,'الشهر الثاني: 1. فعل 2. فعل 3. فعل')


def test_memory_ignores_prior_bot_inventions_across_topics():
    mem=commit_answer(fresh_memory(),CAREER,'اخترع البوت إن عندك مشروع كشف عيوب جاهز')
    mem=commit_answer(mem,'عندي تطبيق نباتات باستخدام CNN','البوت قال عندك اختراع آخر')
    question='نرجع لموضوع شغلي. لخص كل المشاريع اللي أنا ذكرتها في المحادثة دي.'
    sys=build_messages(question,mem)[0]['content']
    assert 'TensorFlow وComputer Vision وLLMs' in sys
    assert 'اخترع البوت' not in sys


def test_rag_five_step_three_error_coverage():
    names={r.name for r in extract_requirements(RAG)}
    assert {'rag_pipeline','rag_five_steps','rag_three_errors'}<=names
    issues=missing_requirements(RAG,'RAG يُحدّث الملفات.\n1. اقرأ الملفات\n2. أجب\nالأخطاء:\n1. نسيان التحديث')
    assert 'rag_steps_incomplete' in issues and 'rag_errors_incomplete' in issues


def test_news_filter_book_is_not_political_breaking_news():
    dt=datetime(2026,11,22,18,19,tzinfo=ZoneInfo('Africa/Cairo'))
    raw=[
      {'url':'https://example.org/book','title':'المسؤولية العالمية تجاه غزة.. كتاب جديد لمركز العودة', 'published':'2026-11-22T17:41:00+02:00','publisher':'Example'},
      {'url':'https://www.reuters.com/world/example','title':'European leaders announce new diplomatic talks','published':'2026-11-22T17:45:00+02:00','publisher':'Reuters'},
      {'url':'https://www.example.net/opinion','title':'Opinion: the new world order','published':'2026-11-22T17:50:00+02:00','publisher':'Opinion'},
    ]
    ev=search_web(NEW,search_fn=lambda _:raw,now=dt)
    assert len(ev.sources)==1 and 'diplomatic talks' in ev.sources[0]['title']
    assert plan_search(NEW,now=dt).start==dt-timedelta(hours=12)


def test_preserve_user_verified_speech_and_stop_modules():
    root=Path(__file__).resolve().parents[1]
    hashes={name:hashlib.sha256((root/'src'/name).read_bytes()).hexdigest() for name in ('speech.py','job_manager.py')}
    assert hashes=={'speech.py':'293a350f1a0451aaf65b6035a860608a0e4bc740492f12f59a726385bee672bf',
                    'job_manager.py':'3eba343fdc94955072599a70f62446c34d7069efdb312b8fb9ab6e3fcfe8a790'}


def test_clean_ui_keeps_full_audio_capture():
    import inspect
    text=inspect.getsource(app.make_ui)
    assert "streaming=False" in text and 'Recovery code' in text and 'إيقاف الرد' in text
    assert 'Stable Mode' not in text and 'السيرفر يولد الإجابة' not in text


def test_chatgpt_shutdown_routed_to_search_and_handles_no_results_safely():
    from src.realtime import needs_fresh_search
    q='قرأت منشورًا بيقول إن شركة OpenAI أعلنت إيقاف ChatGPT بالكامل الأسبوع القادم. هل فيه إعلان رسمي يؤكد ده؟ ابحث عن دليل.'
    assert needs_fresh_search(q)
    now=datetime(2026,11,22,18,20,tzinfo=ZoneInfo('Africa/Cairo'))
    plan=plan_search(q,now=now)
    assert plan.topic=='OpenAI ChatGPT shutdown claim'
    ev=search_web(q,search_fn=lambda _: [],now=now)
    with patch('src.news_quality._search',return_value=[]),patch('app.search_web',side_effect=AssertionError('official query must use official path')),patch('torch.cuda.is_available',return_value=False):
        answer=list(app.turn(q,[],fresh_memory()))[-1][0][-1]['content']
    assert 'غير مُثبت' in answer and 'لم أجد' in answer
    assert 'شائعة كاذبة' not in answer


def test_real_acceptance_suite_has_eight_chat_plus_audio_and_stop():
    from acceptance_suite import CASES, AUDIO_REFERENCE,STOP_PROMPT,AFTER_STOP
    assert len(CASES)==8
    assert [x[0] for x in CASES]==[f'{i:02d}_{name}' for i,name in enumerate(
        ['career','edit','buses','code','memory','rag','news','rumor'],1)]
    assert 'MediaPipe' in AUDIO_REFERENCE
    assert 'MLOps' in STOP_PROMPT and '19 × 23' in AFTER_STOP


def test_acceptance_suite_cannot_pretend_cpu_is_qwen():
    from acceptance_suite import run_ten_case_acceptance
    import pytest
    with patch('torch.cuda.is_available',return_value=False):
        with pytest.raises(RuntimeError,match='CUDA'):
            run_ten_case_acceptance('/tmp/should_not_exist_QA.json')
