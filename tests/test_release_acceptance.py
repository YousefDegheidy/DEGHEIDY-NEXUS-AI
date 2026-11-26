"""User-reported regressions, independent local tests (no GPU/model weights)."""
import contextlib
import copy
import io
import re
import tempfile
import textwrap
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import app
from src.coverage import checklist_for_prompt
from src.dialogue import fresh_memory, choose_budget, build_messages
from src.precision import occupancy_solution, multiple_bus_info, mutable_default_solution
from src.realtime import NewsPlan, WebEvidence, _topic_relevant, OPINION_HEADLINE, plan_search
from src.evidence_guard import compose_coverage_digest

MULTI='عندي 3 أتوبيسات: الأول 40 كرسي و30 راكب، الثاني 35 كرسي و28 راكب، الثالث 25 كرسي و15 راكب. احسب نسبة إشغال كل أتوبيس، ونسبة الإشغال الإجمالية للأسطول، ورتبهم من الأقل للأعلى.'
BUG='راجع الكود ده: def add_item(x, items=[]): items.append(x); return items. اشرح المشكلة واكتب الإصلاح ومعاه 3 اختبارات pytest.'
CAREER='أنا طالب سنة رابعة AI وعندي TensorFlow وPyTorch. اعملي خطة 6 شهور كاملة وأولويات والمدة للوصول Senior.'

class ReleaseAcceptance(unittest.TestCase):
    def test_three_bus_weighted_arithmetic_and_sort(self):
        info=multiple_bus_info(MULTI)
        self.assertEqual([(b['rate'],b['seats'],b['passengers']) for b in info],[(75.,40,30),(80.,35,28),(60.,25,15)])
        ans=occupancy_solution(MULTI)
        for expected in ('75%','80%','60%','73 ÷ 100','73%','الثالث (60%) ← الأول (75%) ← الثاني (80%)'):
            self.assertIn(expected,ans)
        self.assertNotIn('85.71%',ans)
    def test_single_bus_still_correct(self):
        ans=occupancy_solution('أتوبيس فيه 35 كرسي و28 راكب. احسب نسبة الإشغال، واكتب Python function مع 3 اختبارات pytest.')
        self.assertIn('80%',ans)
        self.assertIn('test_regular_bus',ans)
    def test_unparseable_multiple_buses_do_not_use_first_only(self):
        self.assertIsNone(occupancy_solution('عندي 3 أتوبيسات الأول 40 كرسي و30 راكب والثاني 35 كرسي فقط. احسب الإشغال الإجمالي'))
    def test_mutable_default_answer_includes_valid_executable_tests(self):
        answer=mutable_default_solution(BUG)
        self.assertIn('items=None',answer)
        self.assertNotIn('assert add_item(2) == [1, 2]',answer)
        blocks=re.findall(r'```python\n(.*?)\n```',answer,re.S)
        self.assertEqual(len(blocks),2)
        scope={}
        exec(blocks[0]+'\n'+blocks[1],scope)
        for name in ('test_first_call','test_independent_calls','test_supplied_list_is_updated'):
            scope[name]()
    def test_no_forbidden_ui_copy_and_audio_unchanged(self):
        demo=app.make_ui()
        comps=demo.config['components']
        visible=' '.join(str(c.get('props',{}).get(x,'')) for c in comps for x in ('value','label'))
        for forbidden in ('Stable mode','السيرفر يولّد الإجابة','مش بتقص الصوت','سجّل كامل الجملة','الكلام يتحول لنص'):
            self.assertNotIn(forbidden,visible)
        audios=[c for c in comps if c['type']=='audio']
        self.assertEqual(len(audios),1)
        self.assertFalse(audios[0]['props'].get('streaming',True))
        self.assertEqual(audios[0]['props'].get('type'),'numpy')
        self.assertIn('استعادة الرد',visible)
        self.assertIn('إيقاف الرد',visible)
    def test_month_plan_prompt_has_all_months_and_senior_caution(self):
        system=build_messages(CAREER,fresh_memory())[0]['content']
        self.assertIn('months 1 through 6',system)
        self.assertIn('months 5–6',system)
        self.assertIn('several years',system)
        self.assertLessEqual(choose_budget(CAREER),850)
    def test_new_code_uses_deterministic_fix_in_real_turn_cpu(self):
        with patch('torch.cuda.is_available',return_value=False):
            frames=list(app.turn(BUG,[],fresh_memory(),None))
        self.assertIn('test_independent_calls',frames[-1][0][-1]['content'])
        self.assertNotIn('محتاج GPU',frames[-1][0][-1]['content'])
    def test_three_buses_real_turn_cpu(self):
        with patch('torch.cuda.is_available',return_value=False):
            frames=list(app.turn(MULTI,[],fresh_memory(),None))
        self.assertIn('73%',frames[-1][0][-1]['content'])
        self.assertIn('الثالث (60%)',frames[-1][0][-1]['content'])
    def test_opinion_and_astrology_filtered(self):
        now=datetime(2026,11,22,13,tzinfo=ZoneInfo('Africa/Cairo'))
        plan=NewsPlan('أهم الأخبار', 'political news',['politics'],now-timedelta(hours=24),now,False)
        for title in ['Sustained Xi, Soothed Trump, Secured Putin diplomatic Theatre',
                      'Astrological forecast for Ukraine November 23-29',
                      'Opinion: New world order', 'بقلم: التحولات الدولية']:
            self.assertFalse(_topic_relevant({'title':title,'url':'https://news.example.com/a'},plan),title)
    def test_rumor_focused_news_search(self):
        now=datetime(2026,11,22,13,tzinfo=ZoneInfo('Africa/Cairo'))
        plan=plan_search('هل NVIDIA اشترت Microsoft مقابل 5 تريليون دولار النهارده؟',now=now)
        self.assertIn('NVIDIA Microsoft',plan.topic)
        self.assertTrue(any('Nvidia' in q for q in plan.queries))
    def test_news_requires_article_if_detailed_requested_and_has_links(self):
        now=datetime(2026,11,22,13,tzinfo=ZoneInfo('Africa/Cairo'))
        plan=NewsPlan('هات 3 أخبار سياسية', 'political news',['politics'],now-timedelta(hours=24),now,False)
        sources=[{'title':'Title A','url':'https://original.example/a','publisher':'Publisher A','published':'2026-11-22 12:00 EET','date_kind':'published','read_status':'headline_only'},
                 {'title':'Title B','url':'https://original.example/b','article_url':'https://publisher.example/b','publisher':'Publisher B','published':'2026-11-22 12:30 EET','date_kind':'published','read_status':'article_read','article_text':'أعلن مجلس الوزراء عن إجراءات جديدة لتنظيم العبور التجاري وتشديد الإجراءات الخاصة بالموانئ خلال المؤتمر.'}]
        ev=WebEvidence('found','2026-11-22 13:00 EET','Mock',sources,plan=plan)
        ans=compose_coverage_digest('هات 3 أخبار سياسية وملخص من سطرين لكل خبر، ومتكتفيش بالعناوين',ev)
        self.assertIn('وجدت 1 مادة',ans)
        self.assertIn('https://publisher.example/b',ans)
        self.assertNotIn('**Title A**',ans)
    def test_rumor_no_results_is_not_asserted_false(self):
        now=datetime(2026,11,22,13,tzinfo=ZoneInfo('Africa/Cairo'))
        plan=plan_search('سمعت إن NVIDIA اشترت Microsoft مقابل 5 تريليون دولار النهارده، هل ده صحيح؟',now=now)
        ev=WebEvidence('no_results','2026-11-22 13:00 EET','Mock',[],plan=plan)
        with patch('app.should_search',return_value=True),patch('app.search_web',return_value=ev):
            frames=list(app.turn('سمعت إن NVIDIA اشترت Microsoft مقابل 5 تريليون دولار النهارده، هل ده صحيح؟',[],fresh_memory(),None))
        self.assertIn('الادعاء غير مُثبت',frames[-1][0][-1]['content'])
    def test_unchanged_voice_mode_has_hq_model_options(self):
        from src.speech import ASR_OPTIONS
        self.assertTrue(ASR_OPTIONS)
        self.assertTrue(any('turbo' in str(x).lower() or 'medium' in str(x).lower() for x in ASR_OPTIONS))

if __name__=='__main__':unittest.main()
