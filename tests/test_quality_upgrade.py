import unittest
from unittest.mock import patch
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
from src.coverage import extract_requirements,missing_requirements,checklist_for_prompt
from src.dialogue import choose_budget,build_messages,fresh_memory,commit_answer
from src.realtime import plan_search,should_search,search_web,WebEvidence,NewsPlan
from src.evidence_guard import compose_coverage_digest,render_news_response
import app

CAREER=('how can I be a sinior AI engneer في أقل وقت ممكن؟ أنا طالب سنة رابعة AI '
        'وعندي big projects في TensorFlow وPyTorch وComputer Vision وLLMs، '
        'بس خبرتي في الشركات قليلة والـGPA بتاعي مش عالي. أنا مش عايز أحسن GPA دلوقتي، '
        'أنا عايز شغل AI Engineer مدفوع وأوصل لمستوى Senior. '
        'قولي خطة عملية لمدة 6 شهور تناسب مستوايا، وإيه اللي ناقصني، '
        'وقد إيه ممكن أحتاج عشان أبقى job-ready وقد إيه عشان أبقى Senior فعلًا؟')
NEWS=('عايز أهم 3 أخبار سياسية حصلت خلال آخر 24 ساعة بتوقيت القاهرة، '
      'وخصوصًا آخر تطورات ترامب والصين. اقرأ تفاصيل الأخبار من المقالات الأصلية لو متاحة، '
      'وقولي إيه اللي حصل فعلًا وإيه اللي مجرد توقعات أو تصريحات، مع المصدر وتاريخ النشر. '
      'وفيه كلام عن اتفاق سري بين أمريكا والصين بـ9 تريليون دولار، هل ده صحيح ولا إشاعة؟ '
      'وبعدها حلل التأثير المحتمل على الاقتصاد المصري من ناحية قناة السويس والتجارة والطاقة. '
      'افصل الحقائق عن التحليل، ومتخترعش تفاصيل.')
OPS=('فاكر أنا قلتلك إيه في أول كلامنا عن مستوايا في AI والـGPA وهدفي؟ '
     'لخصلي اللي عندي واللي ناقصني من غير ما تخترع حاجة. ودلوقتي عندي project جديد '
     'في Operations Research: أتوبيس فيه 35 seats وفيه 28 passengers. '
     'احسب occupancy rate، واكتب Python function بتحسب نسبة الإشغال لأي أتوبيس '
     'مع 3 unit tests، وقولي إزاي أقدر أستخدم AI وOR لتحسين نظام نقل كامل. اشرح بالمصري وخليك عملي.')
NOW=datetime(2026,11,20,5,4,tzinfo=ZoneInfo('Africa/Cairo'))

def source(title,hour,body=''):
    return {'title':title,'url':'https://example.org/'+str(hour).replace(':',''),
           'article_url':'https://example.org/'+str(hour).replace(':',''),
           'published':NOW.replace(hour=hour,minute=2).strftime('%Y-%m-%d %H:%M EET'),
           'date_kind':'published','publisher':'Example News','read_status':'article_read' if body else 'headline_only',
           'article_text':body,'snippet':''}

class QualityRegression(unittest.TestCase):
    def test_career_never_goes_to_news(self):
        self.assertFalse(should_search(CAREER,fresh_memory()))
    def test_career_task_planner(self):
        reqs={x.name for x in extract_requirements(CAREER)}
        for x in ['career_specific','career_roadmap','career_job_ready','career_senior_timeline','career_gpa','career_gaps']:
            self.assertIn(x,reqs)
    def test_career_budget_large_enough(self):
        self.assertGreaterEqual(choose_budget(CAREER),650)
    def test_career_prompt_has_month_six(self):
        self.assertIn('months 1 through 6',build_messages(CAREER,fresh_memory())[0]['content'])
    def test_bad_career_answer_is_flagged(self):
        ans='الشهر 1: FastAPI. الشهر 2: CI/CD.'
        miss=missing_requirements(CAREER,ans)
        self.assertIn('six_month_roadmap',miss)
        self.assertIn('senior_timeline',miss)
    def test_good_career_answer_not_flagged(self):
        ans="الشهر 1، الشهر 2، الشهر 3، الشهر 4، الشهر 5، الشهر 6. قدم على شغل Junior job-ready خلال 3-9 شهور حسب النتائج. Senior عادة خلال 3-6 سنوات خبرة إنتاجية وليس 6 أشهر."
        self.assertFalse(missing_requirements(CAREER,ans))
    def test_single_pass_even_if_answer_is_incomplete(self):
        calls=[]
        def fake(msgs,budget):
            calls.append(budget)
            yield 'الشهر 1: APIs. الشهر 2: اختبارات.'
        with patch('builtins.print') as log:
            chunks=list(app.fast_guarded_stream(CAREER,fresh_memory(),fake))
        self.assertEqual(len(calls),1)
        self.assertEqual(chunks[-1],'الشهر 1: APIs. الشهر 2: اختبارات.')
        self.assertNotIn('استكمال باقي المطلوب',chunks[-1])
        self.assertTrue(any('COVERAGE REPORT' in str(x) for x in log.call_args_list))
    def test_single_turn_not_double_generated(self):
        calls=[]
        def fake(msgs,budget):calls.append(1);yield 'الفرق بين CNN وRNN هو طريقة معالجة الأنماط والبيانات المتسلسلة.'
        list(app.fast_guarded_stream('اشرح الفرق بين CNN وRNN',fresh_memory(),fake))
        self.assertEqual(len(calls),1)
    def test_news_query_and_tasks(self):
        self.assertTrue(should_search(NEWS,fresh_memory()))
        keys={r.name for r in extract_requirements(NEWS)}
        for x in ['news_current','news_three','news_trump','news_rumor','news_egypt','news_24h']:
            self.assertIn(x,keys)
    def test_exact_rolling_24h_in_cairo(self):
        p=plan_search(NEWS,now=NOW)
        self.assertEqual(p.start,NOW-timedelta(hours=24))
        self.assertEqual(p.end,NOW+timedelta(minutes=1))
        self.assertEqual(p.topic,'mixed political news Trump China')
    def test_no_old_news_in_last_24h(self):
        mock=[{'title':'Trump China summit Tuesday','url':'https://bbc.com/old','published':'2026-11-18T16:00:00+02:00','publisher':'BBC'},
              {'title':'Trump China summit Friday','url':'https://bbc.com/new','published':'2026-11-20T01:00:00+02:00','publisher':'BBC'}]
        ev=search_web(NEWS,search_fn=lambda _:mock,now=NOW)
        self.assertEqual(len(ev.sources),1)
        self.assertIn('Friday',ev.sources[0]['title'])
    def test_news_fallback_coverage_three_rumor_egypt(self):
        plan=plan_search(NEWS,now=NOW)
        fake=[{'title':'Trump China leaders discuss trade','url':'https://example.org/a','published':'2026-11-20T01:00:00+02:00','publisher':'Source A'},
              {'title':'President meets officials to discuss Ukraine','url':'https://example.org/b','published':'2026-11-19T12:00:00+02:00','publisher':'Source B'},
              {'title':'Parliament votes on new budget','url':'https://example.org/c','published':'2026-11-20T02:00:00+02:00','publisher':'Source C'}]
        ev=search_web(NEWS,search_fn=lambda _:fake,now=NOW)
        output=compose_coverage_digest(NEWS,ev)
        self.assertIn('9 تريليون',output)
        self.assertIn('قناة السويس',output)
        self.assertIn('الطاقة',output)
        self.assertIn('الاستثمار',output)
        self.assertIn('ترامب والصين',output)
        self.assertIn('الخبر',output)
        self.assertIn('2026-11-19',output)
        self.assertNotIn('2026-11-18',output)
    def test_news_composite_does_not_return_only_egypt(self):
        now=NOW
        e=search_web(NEWS,search_fn=lambda _: [{'title':'Trump China summit talks','url':'https://example.org/1','publisher':'BBC','published':'2026-11-20T03:00:00+02:00'}],now=now)
        ans,v=render_news_response(NEWS,'فيه أثر اقتصادي محتمل على مصر.',e)
        self.assertIn('9 تريليون',ans)
        self.assertIn('قناة السويس',ans)
        self.assertIn('ترامب والصين',ans)
    def test_earlier_user_memory_retained_for_question_three(self):
        m=commit_answer(fresh_memory(),CAREER,'أنت ذكرت خبرة المشاريع')
        m=commit_answer(m,NEWS,'عنوان مصدّر')
        msg=build_messages(OPS,m)
        full='\n'.join(x['content'] for x in msg)
        self.assertIn('TensorFlow',full)
        self.assertIn('GPA',full)
    def test_ops_requires_code_tests_and_math(self):
        keys={x.name for x in extract_requirements(OPS)}
        self.assertTrue({'memory','occupancy','python','tests','ai_or'}<=keys)
        self.assertGreaterEqual(choose_budget(OPS),650)
    def test_ops_partial_is_caught(self):
        a='أنت طالب سنة رابعة AI وتستخدم TensorFlow وPyTorch، وعندك مشروعات.'
        self.assertTrue({'occupancy_percent','python_code','three_tests'}<=set(missing_requirements(OPS,a)))
    def test_news_cpu_without_llm_still_all_tasks(self):
        ev=search_web(NEWS,search_fn=lambda _: [{'title':'Trump China summit talks','url':'https://example.org/a','publisher':'BBC','published':'2026-11-20T03:00:00+02:00'}],now=NOW)
        with patch('torch.cuda.is_available',return_value=False),patch.object(app,'search_web',return_value=ev),patch.object(app,'enrich_articles',return_value=None):
            frames=list(app.turn(NEWS,[],fresh_memory()))
        ans=frames[-1][0][-1]['content']
        self.assertIn('9 تريليون',ans)
        self.assertIn('قناة السويس',ans)
        self.assertIn('ترامب والصين',ans)

if __name__=='__main__':unittest.main()
