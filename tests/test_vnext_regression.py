import unittest,time,threading
import numpy as np
from unittest.mock import patch
from src import speech
from src.precision import occupancy_info,occupancy_solution
from src.job_manager import ResponseJobManager
from src.dialogue import fresh_memory
import app

SR=speech.SAMPLE_RATE

class VoiceReliability(unittest.TestCase):
    def test_full_audio_amplitude_int16_stereo(self):
        pcm=np.tile(np.array([[6000,7000],[5000,6000]],np.int16),(SR*2//2,1))
        info=speech.audio_diagnostics((SR,pcm))
        self.assertTrue(info['ok'])
        self.assertEqual(info['channels'],2)
        self.assertEqual(info['duration_s'],2.)
    def test_silence_is_visible_to_user_not_sent_to_model(self):
        quiet=np.zeros(4*SR,np.float32)
        self.assertFalse(speech.audio_diagnostics((SR,quiet))['ok'])
        with patch.object(speech,'transcribe_signal',side_effect=AssertionError('model should NOT run')):
            txt,msg=speech.transcription_of_file((SR,quiet),'Original',mode='auto')
        self.assertEqual(txt,'Original')
        self.assertIn('منخفض',msg)
    def test_recording_includes_ending_after_pause(self):
        prefix=np.ones(2*SR,np.float32)*.08
        tail=np.ones(SR,np.float32)*.06
        recording=(SR,np.concatenate([prefix,np.zeros(SR,np.float32),tail]))
        lengths=[]
        def fake_asr(samples):lengths.append(len(samples));return 'أنا يوسف AI and PyTorch'
        txt,status=speech.transcription_of_file(recording,'Existing',fake_asr)
        self.assertEqual(lengths,[4*SR]);self.assertIn('PyTorch',txt)
    def test_gradio_uses_complete_recording_not_streaming_audio(self):
        ui=app.make_ui()
        comps=[c for c in ui.config['components'] if c['type']=='audio']
        self.assertEqual(len(comps),1)
        self.assertFalse(comps[0]['props'].get('streaming',False))
        self.assertEqual(comps[0]['props'].get('type'),'numpy')
    def test_offline_sample_selected_models_without_download(self):
        fake_model=type('Model',(),{'transcribe':lambda self,*args,**kwargs:(
                [type('Seg',(),{'text':' مرحبا AI and PyTorch.'})()],
                type('Info',(),{'language':'ar'})())})()
        sig=np.ones(2*SR,dtype=np.float32)*.02
        with patch.object(speech,'load_asr',return_value=(fake_model,'Mock Whisper')):
            txt=speech.transcribe_signal(sig,mode='auto')
        self.assertEqual(txt,'مرحبا AI and PyTorch.')
    def test_german_stereo_mono_does_not_throw(self):
        pcm=np.ones((48000*2,2),np.float32)*.005
        info=speech.audio_diagnostics((48000,pcm))
        self.assertEqual(info['sample_rate'],48000)
        self.assertEqual(info['channels'],2)

class OccupancyGroundTruth(unittest.TestCase):
    def test_28_of_35_is_80_not_85(self):
        q='أتوبيس فيه 35 كرسي و28 راكب. احسب نسبة الإشغال واكتب Python function ومعاها 3 اختبارات pytest وشرح ML وOR.'
        info=occupancy_info(q);self.assertEqual(info['rate'],80.)
        ans=occupancy_solution(q)
        self.assertIn('80%',ans)
        self.assertIn('test_over_capacity',ans)
        self.assertNotIn('85.71',ans)
    def test_zero_seats_does_not_divide(self):
        self.assertIsNone(occupancy_info('صفر مقاعد و28 راكب'))
    def test_raw_calc_function_behavior(self):
        def calculate_occupancy(passengers,seats):
            if not isinstance(passengers,int) or not isinstance(seats,int):raise TypeError
            if seats<=0 or passengers<0 or passengers>seats:raise ValueError
            return round(100.0*passengers/seats,2)
        self.assertEqual(calculate_occupancy(28,35),80.)
        with self.assertRaises(ValueError):calculate_occupancy(5,0)
        with self.assertRaises(ValueError):calculate_occupancy(36,35)

class StopResponsiveness(unittest.TestCase):
    def test_stop_shows_immediately_and_allows_next_request(self):
        go=threading.Event();started=threading.Event()
        def slow(q,chat,mem,news):
            started.set();go.wait(timeout=2)
            yield [{'role':'user','content':q},{'role':'assistant','content':'late update'}],'',mem,news
        mgr=ResponseJobManager(runner=slow)
        key,_=mgr.submit('long',[],fresh_memory(),None)
        self.assertTrue(started.wait(1))
        t=time.perf_counter();self.assertTrue(mgr.stop(key))
        self.assertLess(time.perf_counter()-t,.3)
        self.assertEqual(mgr.get(key).phase,'cancelled')
        self.assertIn('⏹️',mgr.get(key).snapshot()[0][-1]['content'])
        next_key,_=mgr.submit('short',[],fresh_memory(),None,key)
        self.assertNotEqual(key,next_key)
        go.set()
        for _ in range(100):
            if mgr.get(next_key).phase!='running':break
            time.sleep(.01)
        self.assertEqual(mgr.get(next_key).phase,'finished')
        self.assertEqual(mgr.get(key).phase,'cancelled')

if __name__=='__main__':unittest.main()

class RealisticASRMetricTests(unittest.TestCase):
    def test_exact_transcription_has_zero_word_error(self):
        wave=(SR,np.ones(SR*2,dtype=np.float32)*.06)
        with patch.object(speech,'transcribe_signal',return_value='أنا يوسف واشتغل AI'):
            recognized,report=speech.speech_accuracy_test(wave,'أنا يوسف واشتغل AI','auto')
        self.assertIn('WER=0.0%',report)
        self.assertIn('CER=0.0%',report)
    def test_partial_transcription_has_positive_word_error(self):
        wave=(SR,np.ones(SR*2,dtype=np.float32)*.06)
        with patch.object(speech,'transcribe_signal',return_value='أنا يوسف'):
            _,report=speech.speech_accuracy_test(wave,'أنا يوسف واشتغل AI','auto')
        self.assertIn('WER=50.0%',report)
