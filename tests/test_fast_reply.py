import unittest
from unittest.mock import patch
from types import SimpleNamespace
import app
from src.dialogue import fresh_memory, build_messages, choose_budget
from src.realtime import should_search

Q='أنا طالب في سنة رابعة AI، وعندي مشاريع قوية في TensorFlow وPyTorch وComputer Vision، لكن خبرتي في الشركات محدودة. عايز أشتغل AI Engineer بأسرع وقت. تنصحني بإيه؟'

class TestFastReply(unittest.TestCase):
    def test_plain_career_question_skips_news(self):
        self.assertFalse(should_search(Q, fresh_memory()))

    def test_compact_prompt(self):
        messages=build_messages(Q,fresh_memory())
        self.assertLess(len(messages[0]['content']), 2000)
        self.assertLessEqual(choose_budget(Q), 450)  # avoid clipping normal answers mid-sentence

    def test_normal_chat_one_generation_only(self):
        calls=[]
        def fake_stream(messages,budget):
            calls.append(budget)
            yield 'ممكن تجهز نفسك لوظيفة Junior AI Engineer قريبًا بمشاريع قابلة للنشر، لكن Senior يحتاج سنوات مسؤولية تشغيل فعلية.'
        outputs=list(app.fast_guarded_stream(Q,fresh_memory(),fake_stream))
        self.assertEqual(len(calls),1)
        self.assertTrue(outputs)

    def test_progress_before_first_text(self):
        class Future:
            def __init__(self):self.n=0
            def done(self):
                self.n+=1
                return self.n>=2
            def result(self):return ('fake-proc','fake-model')
        def stream(messages,budget):
            yield 'اعرض 3 مشاريع قوية وتعلم deployment، وخبرة Senior بتحتاج سنوات شغل حقيقي.'
        with patch('torch.cuda.is_available',return_value=True), \
             patch.object(app,'warm_model_in_background',return_value=Future()), \
             patch.object(app,'model_stream',side_effect=stream), \
             patch('app.time.sleep',return_value=None):
            steps=list(app.turn(Q,[],fresh_memory()))
        texts=[step[0][-1]['content'] for step in steps]
        self.assertTrue(any('بجهّز موديل' in t for t in texts))
        self.assertTrue(any('بكتب الإجابة' in t for t in texts))
        self.assertIn('deployment',texts[-1])

if __name__=='__main__':unittest.main()
