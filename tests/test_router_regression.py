import unittest
from unittest.mock import patch
from src.dialogue import fresh_memory,commit_answer
from src.realtime import needs_fresh_search,should_search
import app

QUESTION=("how can I be a sinior AI engneer في أقل وقت ممكن؟ أنا طالب سنة رابعة AI "
"وعندي big projects في TensorFlow وPyTorch وComputer Vision وLLMs "
"بس خبرتي في الشركات قليلة والـGPA بتاعي مش عالي. "
"أنا مش عايز أحسن GPA دلوقتي، أنا عايز شغل AI Engineer مدفوع "
"وأوصل لمستوى Senior. قولي خطة عملية لمدة 6 شهور تناسب مستوايا")

class RouterRegression(unittest.TestCase):
    def test_exact_failure_reproduced_no_more(self):
        self.assertFalse(needs_fresh_search(QUESTION))
        self.assertFalse(should_search(QUESTION,fresh_memory()))
    def test_career_even_after_news_topic_in_history(self):
        m=commit_answer(fresh_memory(),'اخبار ترامب في الصين امبارح','أخبار سياسية')
        self.assertFalse(should_search(QUESTION,m))
        self.assertFalse(should_search('طب أنا عايز شغل AI Engineer دلوقتي، تنصحني بإيه؟',m))
    def test_clear_news_instead(self):
        for q in ('آخر أخبار زيارة ترامب للصين اليوم؟','ايه الاخبار السياسية اللي حصلت امبارح؟',
                  'ترامب راح الصين واتكلموا في ايه','What are the latest political news today?',
                  'Xi and Trump met today?'):
            with self.subTest(q=q): self.assertTrue(should_search(q,fresh_memory()))
    def test_general_words_cannot_trigger_news(self):
        for q in ('أنا عايز أتعلم Python دلوقتي','how to become a senior AI engineer today',
                  'حاليًا عندي مشاريع كبيرة في computer vision','دراسة السياسة والمجتمع بتفيدني إزاي؟',
                  'أنا خبرتي قليلة عايز اشتغل دلوقتي','عايز أعمل CV وأبدأ تقديم شغل حاليًا',
                  'أخبارك إيه يا معلم؟','what is a presidential system in politics?'):
            with self.subTest(q=q): self.assertFalse(should_search(q,fresh_memory()))
    def test_news_followup_remains_when_relevant(self):
        m=commit_answer(fresh_memory(),'اخبار زيارة ترامب للصين امبارح','وردت تغطية سياسية')
        self.assertTrue(should_search('طب ترامب اتفق على ايه؟',m))
        self.assertFalse(should_search('طب ده هيأثر على مصر إزاي؟',m))
        self.assertFalse(should_search('طب أنا عايز اتعلم TensorFlow دلوقتي',m))
    def test_chat_generation_path_not_search(self):
        # Mock GPU-model generation: the router should never access search_web.
        with patch('torch.cuda.is_available',return_value=True), \
             patch.object(app,'search_web',side_effect=AssertionError('NEWS ROUTE CALLED')), \
             patch.object(app,'warm_model_in_background',return_value=None), \
             patch.object(app,'fast_guarded_stream',return_value=iter(['ابدأ ببناء مشاريع قابلة للنشر ثم قدم على وظائف Junior AI Engineer.'])):
            frames=list(app.turn(QUESTION,[],fresh_memory(),None))
        self.assertIn('Junior AI Engineer',frames[-1][0][-1]['content'])
        self.assertNotIn('أخبار',frames[-1][0][-1]['content'])
    def test_news_state_is_not_reused_after_topic_reset(self):
        m=commit_answer(fresh_memory(),'اخبار ترامب الصين امبارح','تفاصيل')
        fake={'old':'news'}
        with patch('torch.cuda.is_available',return_value=True), \
             patch.object(app,'search_web',side_effect=AssertionError('NO NEWS')), \
             patch.object(app,'warm_model_in_background',return_value=None), \
             patch.object(app,'fast_guarded_stream',return_value=iter(['ابدأ بالـPortfolio.'])):
            frames=list(app.turn(QUESTION,[],m,fake))
        self.assertIsNone(frames[-1][3])
        self.assertIn('Portfolio',frames[-1][0][-1]['content'])

if __name__=='__main__': unittest.main()
