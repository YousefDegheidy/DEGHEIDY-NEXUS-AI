"""Three exact user prompts through the chat pipeline with MOCKED LLM/news.
Do not claim these run Qwen3.5 weights or validate external events.
"""
import unittest
from unittest.mock import patch
from src.dialogue import fresh_memory
from src.realtime import search_web
from src.coverage import missing_requirements
from tests.test_quality_upgrade import CAREER, NEWS, OPS, NOW
import app

class MockThreeQuestionPipeline(unittest.TestCase):
    def test_all_three_turns_with_one_session(self):
        messages=[];memory=fresh_memory();news_state=None
        answers={
          'career':('الشهر 1: API وتقييم. الشهر 2: اختبارات. الشهر 3: النشر. '
                    'الشهر 4: monitoring. الشهر 5: مقابلات. الشهر 6: تحسين البورتفوليو. '
                    'ابدأ تقديم شغل Junior AI Engineer مدفوع؛ الوصول إلى job-ready قد يستغرق 3-9 أشهر '
                    'بحسب قوة المشاريع والمقابلات. منصب Senior عادة يحتاج 3-6 سنوات خبرة production ومسؤولية فعلية.'),
          'ops':('أنت قلت إنك سنة رابعة AI وعندك TensorFlow وPyTorch ومشاريع كبيرة وGPA منخفض '
                 'وهدفك شغل AI Engineer. لم تقل إنك تفتقد Git أو DevOps؛ ده محتاج تقييم. '
                 'نسبة الإشغال = 28/35*100 = 80%. '
                 '```python\ndef occupancy(n, capacity):\n    if capacity <= 0: raise ValueError("capacity")\n    return 100*n/capacity\n\ndef test_full(): assert occupancy(35,35) == 100\n\ndef test_partial(): assert occupancy(28,35) == 80\n\ndef test_empty(): assert occupancy(0,35) == 0\n```'
                 'استخدم AI للتنبؤ بالطلب وOR لتخصيص الحافلات والخطوط تحت قيود السعة والتكلفة.')
        }
        def generator(msg,budget):
            last=msg[-1]['content']
            if 'فاكر أنا قلتلك' in last:
                yield answers['ops']
            elif 'how can I be' in last:
                yield answers['career']
            else:
                yield 'إجابة قصيرة.'
        with patch('torch.cuda.is_available',return_value=True), \
             patch.object(app,'warm_model_in_background',return_value=None), \
             patch.object(app,'model_stream',side_effect=generator), \
             patch.object(app,'enrich_articles',return_value=None):
            for question in [CAREER,NEWS,OPS]:
                if question==NEWS:
                    fake=[{'title':'Trump China summit talks', 'url':'https://example.org/a',
                           'published':'2026-11-20T03:00:00+02:00','publisher':'BBC'},
                          {'title':'Parliament election announcement','url':'https://example.org/b',
                           'published':'2026-11-19T12:00:00+02:00','publisher':'Reuters'},
                          {'title':'President meets diplomats','url':'https://example.org/c',
                           'published':'2026-11-19T16:00:00+02:00','publisher':'AP News'}]
                    evidence=search_web(NEWS,now=NOW,search_fn=lambda _:fake)
                    with patch.object(app,'search_web',return_value=evidence):
                        snaps=list(app.turn(question,messages,memory,news_state))
                else:
                    snaps=list(app.turn(question,messages,memory,news_state))
                messages,_,memory,news_state=snaps[-1]
                result=messages[-1]['content']
                self.assertFalse(missing_requirements(question,result),
                                 f"Missing in turn: {question[:40]} => {missing_requirements(question,result)}")
                if question==NEWS:
                    self.assertIn('9 تريليون',result)
                    self.assertIn('قناة السويس',result)
                    self.assertNotIn('2026-11-18',result)
                if question==OPS:
                    self.assertIn('28/35',result)
                    self.assertIn('لم تقل',result)
        self.assertEqual(len(memory['topics']),2)  # career and OR can relate; news separate

if __name__=='__main__':unittest.main()
