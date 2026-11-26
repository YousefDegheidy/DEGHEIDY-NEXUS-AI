"""Connection-stable UI behavioral tests, without loading Qwen or network access."""
import threading
import time
import unittest
from unittest.mock import patch

from src.job_manager import ResponseJobManager
from src.dialogue import fresh_memory
from src.realtime import plan_search, search_web
from zoneinfo import ZoneInfo
from datetime import datetime
import app

class StableModeTests(unittest.TestCase):
    def test_poll_can_recover_finished_response_after_client_stops_listening(self):
        started=threading.Event()
        proceed=threading.Event()
        def slow_turn(q, chat, memory, news):
            yield [{'role':'user','content':q}, {'role':'assistant','content':'بدأ الرد'}], '', memory,news
            started.set()
            assert proceed.wait(timeout=3)
            yield [{'role':'user','content':q}, {'role':'assistant','content':'الإجابة اكتملت بالكامل'}], '', memory,news
        manager=ResponseJobManager(runner=slow_turn)
        key,_=manager.submit('سؤال طويل',[],fresh_memory(),None)
        self.assertTrue(started.wait(2))
        self.assertEqual(manager.get(key).snapshot()[0][-1]['content'],'بدأ الرد')
        # No browser polling at all during backend completion.
        proceed.set()
        for _ in range(100):
            if manager.get(key).phase!='running':break
            time.sleep(.01)
        self.assertEqual(manager.get(key).phase,'finished')
        self.assertEqual(manager.get(key).snapshot()[0][-1]['content'],'الإجابة اكتملت بالكامل')
        # Same unguessable recovery key reconnects to the same state.
        self.assertIsNotNone(manager.get(key))
        self.assertIsNone(manager.get('not-the-key'))

    def test_no_new_job_while_current_generation_running(self):
        proceed=threading.Event()
        def slow_turn(q,chat,m,n):
            assert proceed.wait(3)
            yield [{'role':'user','content':q},{'role':'assistant','content':'done'}],'',m,n
        manager=ResponseJobManager(runner=slow_turn)
        key,_=manager.submit('question1',[],fresh_memory(),None)
        key2,notice=manager.submit('question2',[],fresh_memory(),None,key)
        self.assertEqual(key,key2)
        self.assertIn('شغال',notice)
        proceed.set()

    def test_cancel_discards_late_changes(self):
        hold=threading.Event()
        def slow_turn(q,chat,m,n):
            yield [{'role':'user','content':q},{'role':'assistant','content':'جزء أول'}],'',m,n
            hold.wait(2)
            yield [{'role':'user','content':q},{'role':'assistant','content':'جزء تاني بعد الإيقاف'}],'',m,n
        manager=ResponseJobManager(runner=slow_turn)
        key,_=manager.submit('long question',[],fresh_memory(),None)
        for _ in range(60):
            if manager.get(key).snapshot()[0][-1]['content']=='جزء أول':break
            time.sleep(.01)
        self.assertTrue(manager.stop(key))
        hold.set()
        for _ in range(70):
            if manager.get(key).phase!='running':break
            time.sleep(.01)
        self.assertEqual(manager.get(key).phase,'cancelled')
        self.assertNotIn('جزء تاني',manager.get(key).snapshot()[0][-1]['content'])

    def test_ui_timer_and_short_polling_callbacks(self):
        ui=app.make_ui()
        self.assertTrue(any(type(b).__name__=='Timer' for b in ui.blocks.values()))
        dependencies=ui.config['dependencies']
        self.assertGreaterEqual(sum(d.get('queue') is False for d in dependencies),5)
        self.assertTrue(any('tick' in str(d.get('targets')) for d in dependencies))

    def test_news_excludes_editorial_not_just_filters_by_date(self):
        now=datetime(2026,11,21,17,50,tzinfo=ZoneInfo('Africa/Cairo'))
        q='هات أهم 3 أخبار سياسية حصلت خلال آخر 24 ساعة بتوقيت القاهرة'
        src=[
          {'title':'كتب/ د. سلطان مشعل: الحضور الجيوسياسي لمصر.. حين تكون الدولة أكبر من حدودها','url':'https://almjhar.com/analysis','published':'2026-11-21T16:29:00+02:00','publisher':'Opinion'},
          {'title':'المجلس الأوروبي يناقش مشروع الميزانية الجديدة','url':'https://reuters.com/budget','published':'2026-11-21T15:00:00+02:00','publisher':'Reuters'},
        ]
        found=search_web(q,search_fn=lambda _:src,now=now)
        self.assertEqual(len(found.sources),1)
        self.assertIn('المجلس الأوروبي',found.sources[0]['title'])

    def test_clear_stops_prior_session_job(self):
        cancelled=[]
        with patch.object(app._UI_JOBS,'stop',side_effect=lambda k:cancelled.append(k)):
            vals=app._clear_background('tok')
        self.assertEqual(cancelled,['tok'])
        self.assertEqual(len(vals),9)

if __name__=='__main__':unittest.main()
