"""Tests for uninterrupted single-pass responses. No GPU or model weights needed."""
import unittest
from unittest.mock import patch
import app
from src.dialogue import choose_budget, fresh_memory
from src.coverage import missing_requirements
from tests.test_quality_upgrade import CAREER, OPS

class SmoothStreamTests(unittest.TestCase):
    def test_no_second_generation_after_missing_months(self):
        calls=[]
        def mock_model(messages,budget):
            calls.append((messages,budget))
            yield 'الشهر 1: API.'
            yield 'الشهر 1: API. الشهر 2: Docker.'
        out=list(app.fast_guarded_stream(CAREER,fresh_memory(),mock_model))
        self.assertEqual(len(calls),1)
        self.assertEqual(len(out),2)
        self.assertFalse(any('استكمال باقي المطلوب' in x for x in out))
    def test_complete_long_output_one_generation(self):
        answer=('الشهر 1: APIs. الشهر 2: Docker. الشهر 3: النشر. '
            'الشهر 4: مراقبة. الشهر 5: مقابلات. الشهر 6: مشاريع. '
            'الاستعداد لـJunior job-ready قد يحتاج 3-9 شهور حسب التقييم؛ '
            'Senior عادة يحتاج 3-6 سنوات خبرة production حقيقية.')
        calls=[]
        def mock_model(msgs,budget):
            calls.append(budget)
            for i in range(1,len(answer)+1,17):yield answer[:i+17]
        out=list(app.fast_guarded_stream(CAREER,fresh_memory(),mock_model))
        self.assertEqual(len(calls),1)
        self.assertEqual(out[-1],answer)
        self.assertFalse(missing_requirements(CAREER,out[-1]))
    def test_budget_preserves_code_and_multi_month_plan(self):
        self.assertGreaterEqual(choose_budget(CAREER),700)
        self.assertGreaterEqual(choose_budget(OPS),800)
        self.assertLessEqual(choose_budget(CAREER),850)
        self.assertLessEqual(choose_budget(OPS),950)
    def test_actual_chat_turn_calls_stream_once(self):
        calls=[]
        def fake(msgs,budget):
            calls.append(budget)
            yield 'الشهر 1: APIs. الشهر 2: Docker.'
        with patch('torch.cuda.is_available',return_value=True), \
             patch.object(app,'warm_model_in_background',return_value=None), \
             patch.object(app,'model_stream',side_effect=fake):
            frames=list(app.turn(CAREER,[],fresh_memory()))
        self.assertEqual(len(calls),1)
        self.assertNotIn('استكمال باقي المطلوب',frames[-1][0][-1]['content'])

if __name__=='__main__':unittest.main()
