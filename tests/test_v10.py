import sys, unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import speech
from src.dialogue import fresh_memory, commit_answer, build_messages, evaluate_reply

SR=speech.SAMPLE_RATE

def sine(d, freq=220):
    n=int(d*SR)
    return (.25*np.sin(2*np.pi*freq*np.arange(n)/SR)).astype(np.float32)


def simple_vad(frame):
    return float(np.sqrt(np.mean(frame*frame)))>.04


class VoiceTests(unittest.TestCase):
    def setUp(self):
        self.calls=[]
        def recog(audio):
            self.calls.append(audio.copy())
            return 'اسمي يوسف and I want to be a Senior AI Engineer'
        self.recog=recog

    def feed(self, st, arr):
        text,st,status=speech.ingest_chunk((SR,arr),st,self.recog,simple_vad)
        return text,st,status

    def test_silence_never_inferred_and_repeated_silence_counted(self):
        st=speech.fresh_voice()
        blank=np.zeros(int(.99*SR),np.float32)
        for _ in range(4):
            _,st,_=self.feed(st,blank)
        self.assertEqual(len(self.calls),0)
        self.assertEqual(speech.voice_view(st),'')
        self.assertGreater(st['processed_s'],3.9)

    def test_single_utterance_committed_once_after_silence(self):
        st=speech.fresh_voice('Previous question')
        _,st,_=self.feed(st,sine(1.5))
        _,st,_=self.feed(st,sine(1.5))
        _,st,_=self.feed(st,np.zeros(int(1.2*SR),np.float32))
        self.assertIn('Senior AI Engineer',speech.voice_view(st))
        self.assertEqual(speech.voice_view(st).count('Senior AI Engineer'),1)
        before=len(self.calls)
        _,st,_=self.feed(st,np.zeros(int(1.2*SR),np.float32))
        self.assertEqual(len(self.calls),before)
        self.assertEqual(st['confirmed'].count('Senior AI Engineer'),1)

    def test_preview_replaced_and_then_committed_not_appended(self):
        st=speech.fresh_voice()
        text,st,_=self.feed(st,sine(1.8))
        self.assertEqual(text,'')
        text,st,_=self.feed(st,sine(1.1,230))
        self.assertIn('I want',text) # preview
        text,st,_=self.feed(st,np.zeros(int(1.1*SR),np.float32))
        self.assertEqual(text.count('I want'),1)
        self.assertEqual(st['preview'],'')

    def test_identical_non_silent_stream_chunk_ignored(self):
        st=speech.fresh_voice()
        chunk=sine(.99)
        _,st,_=self.feed(st,chunk)
        n=st['processed_s']
        _,st,_=self.feed(st,chunk)
        self.assertAlmostEqual(st['processed_s'],n)

    def test_repeated_real_speech_not_silently_removed(self):
        st=speech.fresh_voice()
        for f in (220,320):
            _,st,_=self.feed(st,sine(1.0,f))
            _,st,_=self.feed(st,np.zeros(int(1.1*SR),np.float32))
        self.assertEqual(st['confirmed'].count('I want'),2)

    def test_stop_is_idempotent(self):
        st=speech.fresh_voice()
        _,st,_=self.feed(st,sine(1.0))
        v,st,_=speech.finalize_audio(st,self.recog)
        v2,st,_=speech.finalize_audio(st,self.recog)
        self.assertEqual(v,v2)
        self.assertEqual(len(self.calls),1)

    def test_normalize_stereo_int16(self):
        raw=np.array([[1000,-1000],[3000,3000]],dtype=np.int16)
        out=speech.normalize_audio((SR,raw))
        self.assertEqual(len(out),2)
        self.assertAlmostEqual(float(out[0]),0.,places=4)
        self.assertAlmostEqual(float(out[1]),3000/32768,places=4)

    def test_uploaded_audio_preserves_original_text(self):
        text,log=speech.transcription_of_file((SR,sine(2)), 'Old', self.recog)
        self.assertIn('Old اسمي يوسف',text)

    def test_cpu_mode_doesnt_load_chat(self):
        from app import make_ui
        ui=make_ui()
        self.assertGreater(len(ui.fns),3)
        names=[x.get('label','') for x in ui.config['components'] if x.get('type')=='textbox' for x in [x.get('props',{})]]
        self.assertIn('Ask ??',names)
        self.assertNotIn('Language',str(ui.config))

    def test_multilingual_asr_request_is_transcribe_not_translate(self):
        import inspect
        source=inspect.getsource(speech.transcribe_signal)
        self.assertIn("task='transcribe'",source)
        self.assertIn('language=None',source)
        self.assertIn('condition_on_previous_text=False',source)

    def test_quality_checks_missing_timeline(self):
        memory=fresh_memory()
        q='how to be a sinior in aai و ممكن ياخد مني وقت قد اه؟'
        issues=evaluate_reply(q,'اتعلم Python وحضر مؤتمرات.',memory)
        self.assertIn('missing_or_ambiguous_time',issues)

    def test_followup_retains_role_goal(self):
        q='how to be a sinior in aai في وقت قليل'
        mem=commit_answer(fresh_memory(),q,'عادةً سنوات من الخبرة')
        msg=build_messages('طب لو عندي مشارييع كبيرة هقلل المده قد ايه؟',mem)
        self.assertIn(q,msg[0]['content'])


if __name__=='__main__':
    unittest.main()

class DialogueQualityTests(unittest.TestCase):
    def test_unwarranted_senior_months(self):
        q='how to be a senior in AI في وقت قليل'
        wrong='تقدر تصبح Senior AI Engineer خلال 18 إلى 24 شهر لو بتتعلم.'
        self.assertIn('implausible_seniority_time',evaluate_reply(q,wrong,fresh_memory()))

    def test_internal_checklist_leak_flagged(self):
        q='I want career advice'
        answer='**السؤال**: career advice\n✅ ACTION: اتعلم كويس'
        self.assertIn('internal_checklist_leak',evaluate_reply(q,answer,fresh_memory()))

    def test_long_senior_timeline_not_flagged_for_months(self):
        q='how to be a senior in AI في وقت قليل'
        good='ده عادة مسار عدة سنوات من الخبرة الفعلية، وقد يستغرق 3 إلى 6 سنوات حسب المؤسسة.'
        self.assertNotIn('implausible_seniority_time',evaluate_reply(q,good,fresh_memory()))
