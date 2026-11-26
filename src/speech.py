"""Voice dictation: zero audio overlap at commit, silence-triggered finalization.

- Faster Whisper large-v3 is default with GPU; small is default without GPU.
- Transcribe, never translate; no manual language selection.
- WebRTC VAD gates inference and prevents ASR-on-silence hallucinations.
- A segment is decoded several times for PREVIEW, but appended ONCE when committed.
- Both Arabic/English code switching and accuracy remain empirical questions.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import ctypes
from functools import lru_cache
from typing import Callable

import numpy as np

SAMPLE_RATE = 16000
FRAME_SAMPLES = 480  # WebRTC VAD: exactly 30ms at 16kHz
PREVIEW_MIN_SECONDS = 2.0
PREVIEW_EVERY_SECONDS = 2.6
SILENCE_FINALIZE_SECONDS = 0.9
MAX_SEGMENT_SECONDS = 23.0
PRE_ROLL_SECONDS = .24
ASR_LOCK = threading.Lock()


def fresh_voice(base: str = '') -> dict:
    return dict(base=(base or '').strip(), confirmed='', preview='',
                segment=np.empty(0, dtype=np.float32), frame_buffer=np.empty(0, dtype=np.float32),
                preroll=np.empty(0, dtype=np.float32), seen_voice=False, voiced_frames=0,
                silent_frames=0, processed_s=0.0, preview_at=0.0,
                last_chunk_hash='', stopped=False, status='🎙️ جاهز للتسجيل — Arabic + English تلقائيًا.')


def normalize_audio(audio) -> np.ndarray:
    if audio is None:
        return np.empty(0, np.float32)
    sr, raw = audio
    y = np.asarray(raw)
    source_dtype = y.dtype
    if y.size == 0:
        return np.empty(0, np.float32)
    if y.ndim == 2:
        y = y.astype(np.float32).mean(axis=1)
    if np.issubdtype(source_dtype, np.integer):
        # Crucial: normalize integers BEFORE resampling or VAD.
        y = y.astype(np.float32) / max(abs(np.iinfo(source_dtype).min), np.iinfo(source_dtype).max)
    else:
        y = y.astype(np.float32)
    y = np.nan_to_num(y, copy=False, nan=0, posinf=0, neginf=0)
    if int(sr) != SAMPLE_RATE:
        import librosa
        y = librosa.resample(y, orig_sr=int(sr), target_sr=SAMPLE_RATE)
    return np.clip(y, -1, 1).astype(np.float32)


def voice_view(st: dict) -> str:
    return ' '.join(x for x in [st.get('base', '').strip(), st.get('confirmed', '').strip(),
                                st.get('preview', '').strip()] if x).strip()


def append_text(left: str, right: str) -> str:
    # No word-based de-duplication. Legitimate repetition must remain visible.
    return ' '.join(x for x in [left.strip(), right.strip()] if x)


def same_chunk(chunk: np.ndarray, st: dict) -> bool:
    """Reject exact repeated stream payload, not genuine repeated words."""
    sig = hashlib.blake2b(chunk.tobytes(), digest_size=12).hexdigest()
    # Silent chunks may legitimately be bitwise-identical; keep them for VAD timing.
    is_duplicate = (sig == st.get('last_chunk_hash', '') and
                    float(np.sqrt(np.mean(chunk * chunk))) >= .005)
    st['last_chunk_hash'] = sig
    return is_duplicate


def has_voice(samples: np.ndarray) -> bool:
    if samples.size != FRAME_SAMPLES:
        return False
    # Conservative first-pass energy gate plus WebRTC VAD.
    rms = float(np.sqrt(np.mean(samples * samples)))
    if rms < .002:
        return False
    try:
        import webrtcvad
        pcm = (np.clip(samples, -1, 1) * 32767).astype('<i2').tobytes()
        return bool(_get_vad().is_speech(pcm, SAMPLE_RATE))
    except ImportError:
        # Offline test fallback only. Install webrtcvad-wheels on Colab.
        return rms >= .014


@lru_cache(maxsize=1)
def _get_vad():
    import webrtcvad
    return webrtcvad.Vad(1)



# Use the COMPLETE captured microphone recording by default. In streaming mode,
# Gradio's stop event may race with the last chunk or a VAD silence boundary.
# The old live implementation is retained for compatibility but NOT the default UI.
ASR_OPTIONS = [
    ('تلقائي (GPU Turbo إن توفرت الذاكرة، وإلا CPU Medium)', 'auto'),
    ('أعلى دقة على GPU: Whisper Large-v3 (أبطأ)', 'accurate'),
    ('دقة أعلى على CPU: Whisper Medium (أبطأ)', 'medium'),
    ('سريع على CPU: Whisper Small (أقل دقة)', 'small'),
]


def audio_diagnostics(audio) -> dict:
    """Inspect the received waveform, no model or network needed."""
    if audio is None:
        return {'ok':False,'reason':'المتصفح لم يرسل تسجيلًا. افحص إذن الميكروفون واختر جهاز الإدخال الصحيح.'}
    try:
        sr, samples = audio
        original = np.asarray(samples)
        raw_len = original.shape[0] if original.ndim else 0
        signal = normalize_audio(audio)
        if not raw_len or signal.size==0:
            return {'ok':False,'reason':'وصل ملف صوتي فارغ.'}
        rms=float(np.sqrt(np.mean(signal*signal)))
        peak=float(np.max(np.abs(signal)))
        duration=round(len(signal)/SAMPLE_RATE,2)
        # Low RMS is diagnostic only: don't drop quiet speech based on volume.
        ok = duration >= .7 and rms >= .00030
        reason = ('التسجيل قصير جدًا؛ سجل 4–10 ثوانٍ على الأقل.' if duration<.7 else
                  'الصوت منخفض جدًا. ارفع مستوى الميكروفون وراجع اختيار جهاز الإدخال.' if rms<.00030 else
                  'الصوت وصل للسيرفر بالكامل؛ تقدر تشغّله من مشغل الصوت قبل تحويله.')
        return {'ok':ok,'duration_s':duration,'sample_rate':int(sr),
                'channels':int(original.shape[1]) if original.ndim==2 else 1,
                'rms':round(rms,5),'peak':round(peak,5),
                'clipping':bool(np.mean(np.abs(signal)>.995)>.03),
                'reason':reason}
    except Exception as exc:
        return {'ok':False,'reason':f'تعذّر قراءة التسجيل: {type(exc).__name__}: {str(exc)[:90]}'}


def describe_audio(audio):
    data=audio_diagnostics(audio)
    if not data['ok']:
        return '⚠️ '+data['reason'] + (f" (المدة {data['duration_s']} ثانية)" if 'duration_s' in data else '')
    return (f"🎤 التسجيل وصل: {data['duration_s']} ثانية؛ معدل العينات الأصلي {data['sample_rate']} Hz؛ "
            f"{data['channels']} قناة؛ RMS={data['rms']}؛ Peak={data['peak']}. "
            + ('⚠️ يوجد تقطيع صوتي بسبب ارتفاع المستوى.' if data['clipping'] else '✅ '+data['reason']))


def _gpu_asr_ready(required_bytes=4*1024**3):
    try:
        import torch
        if not torch.cuda.is_available() or os.getenv('DEGHEIDY_ASR_CPU_ONLY')=='1':
            return False
        ctypes.CDLL('libcublas.so.12')
        free,_=torch.cuda.mem_get_info()
        return free>=required_bytes
    except Exception:
        return False


@lru_cache(maxsize=4)
def _load_by_profile(mode):
    from faster_whisper import WhisperModel
    import torch
    mode=(mode or 'auto').lower().strip()
    if mode=='small':
        candidates=[('small','cpu','int8')]
    elif mode=='medium':
        candidates=[('medium','cpu','int8'),('small','cpu','int8')]
    elif mode=='accurate':
        candidates=([('large-v3','cuda','int8_float16')] if _gpu_asr_ready(6*1024**3) else [])
        candidates += [('medium','cpu','int8'),('small','cpu','int8')]
    else:
        candidates=([('large-v3-turbo','cuda','int8_float16')] if _gpu_asr_ready() else [])
        candidates += [('medium','cpu','int8'),('small','cpu','int8')]
    errors=[]
    for model_id,device,compute in candidates:
        try:
            print(f'[ASR] loading {model_id} on {device} ({compute})',flush=True)
            model=WhisperModel(model_id,device=device,compute_type=compute,
                               cpu_threads=4 if device=='cpu' else 0)
            return model,f'Whisper {model_id} ({device})'
        except Exception as exc:
            errors.append(f'{model_id}/{device}: {type(exc).__name__}: {str(exc)[:80]}')
            print('[ASR FALLBACK]',errors[-1],flush=True)
            if device=='cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
    raise RuntimeError('Cannot load ASR model; '+'; '.join(errors))


def load_asr(mode='auto'):
    """Compatible wrapper. Does NOT download Qwen and won't evict chat weights."""
    return _load_by_profile(mode)


def transcribe_signal(signal: np.ndarray, mode='auto') -> str:
    """Whole utterance speech recognition. NEVER remove words using per-chunk VAD."""
    signal = np.asarray(signal,dtype=np.float32)
    if signal.size < SAMPLE_RATE*.7 or float(np.sqrt(np.mean(signal**2)))<.0003:
        return ''
    with ASR_LOCK:
        for attempt in range(2):
            try:
                model,description=load_asr(mode)
                print('[ASR] inference length_s=',round(len(signal)/SAMPLE_RATE,1),
                      'model=',description,flush=True)
                t0=__import__('time').perf_counter()
                segments,info=model.transcribe(
                    signal,task='transcribe',language=None,beam_size=5,
                    temperature=0,condition_on_previous_text=False,
                    vad_filter=False,multilingual=True,
                    no_speech_threshold=.7,compression_ratio_threshold=2.4,
                    log_prob_threshold=-1.0,
                    initial_prompt='The speaker may switch between Arabic and English and mention AI, '
                                   'Python, TensorFlow, PyTorch, MediaPipe, OpenCV, FastAPI, RAG and Chatbot.'
                )
                texts=[re.sub(r'\s+',' ',s.text).strip() for s in segments]
                result=' '.join(t for t in texts if t)
                print('[ASR] took_s=',round(__import__('time').perf_counter()-t0,2),
                      '| detected_language=',getattr(info,'language','?'),flush=True)
                return result
            except (RuntimeError, OSError) as exc:
                msg=str(exc).lower()
                if attempt==0 and mode not in ('medium','small') and any(
                     x in msg for x in ('cublas','cudnn','cuda','out of memory','ct2')):
                    print('[ASR] GPU failed, retrying CPU Medium:',str(exc)[:180],flush=True)
                    mode='medium';continue
                raise
    return ''

def _commit(st: dict, recognizer: Callable[[np.ndarray], str]):
    """Finalize unique UNCOMMITTED speech once; never carry acoustic overlap."""
    if st['seen_voice'] and len(st['segment']) >= SAMPLE_RATE * .40:
        final = recognizer(st['segment'])
        if final:
            st['confirmed'] = append_text(st['confirmed'], final)
    st['segment'] = np.empty(0, np.float32)
    st['preview'] = ''
    st['seen_voice'] = False
    st['voiced_frames'] = 0
    st['silent_frames'] = 0
    st['preview_at'] = 0.0


def ingest_chunk(audio, state: dict, recognizer=transcribe_signal,
                 vad_checker=has_voice):
    """Called by Gradio Audio.stream. Return (editable text, state, status).

    Voice activity is checked on 30ms frames, with acoustic pre-roll.
    Preview is REPLACED, not appended; committed audio is never decoded again.
    """
    st = state if state is not None else fresh_voice()
    if st.get('stopped'):
        return voice_view(st), st, st.get('status', '')
    chunk = normalize_audio(audio)
    if not len(chunk) or same_chunk(chunk, st):
        return voice_view(st), st, st.get('status', '')
    st['frame_buffer'] = np.concatenate([st['frame_buffer'], chunk])
    while len(st['frame_buffer']) >= FRAME_SAMPLES:
        frame = st['frame_buffer'][:FRAME_SAMPLES].copy()
        st['frame_buffer'] = st['frame_buffer'][FRAME_SAMPLES:]
        speaking = vad_checker(frame)
        if not st['seen_voice']:
            st['preroll'] = np.concatenate((st['preroll'], frame))[-int(PRE_ROLL_SECONDS*SAMPLE_RATE):]
            if speaking:
                st['seen_voice'] = True
                st['segment'] = st['preroll'].copy()
                st['preroll'] = np.empty(0, np.float32)
                st['silent_frames'] = 0
                st['voiced_frames'] = 1
        else:
            st['segment'] = np.concatenate((st['segment'], frame))
            st['voiced_frames'] += int(speaking)
            st['silent_frames'] = 0 if speaking else st['silent_frames'] + 1
            should_finish = (st['silent_frames'] * .03 >= SILENCE_FINALIZE_SECONDS or
                             len(st['segment']) / SAMPLE_RATE >= MAX_SEGMENT_SECONDS)
            if should_finish:
                try:
                    _commit(st, recognizer)
                    st['status'] = '✅ خلصت جملة؛ اتثبتت في النص. كمل كلامك.'
                except Exception as exc:
                    st['status'] = f'تعذر تفريغ آخر مقطع: {type(exc).__name__}: {str(exc)[:100]}'
                    st['preview'] = ''
                    # Keep the speech until the user stops to allow retry.
                    st['silent_frames'] = 0
        st['processed_s'] += .03
    seconds = len(st['segment']) / SAMPLE_RATE
    if (st['seen_voice'] and seconds >= PREVIEW_MIN_SECONDS and
            seconds - st['preview_at'] >= PREVIEW_EVERY_SECONDS):
        try:
            st['preview'] = recognizer(st['segment'])
            st['preview_at'] = seconds
            st['status'] = '🎙️ بكتب أثناء الكلام — النص المؤقت ممكن يتصحح لحد ما تسكت.'
        except Exception as exc:
            st['status'] = f'مشكلة في التفريغ: {type(exc).__name__}: {str(exc)[:110]}'
    elif not st['seen_voice'] and not st['confirmed']:
        st['status'] = '🎙️ بسمعك... اتكلم قريب من الميكروفون.'
    return voice_view(st), st, st.get('status', '')


def finalize_audio(state: dict, recognizer=transcribe_signal):
    st = state if state is not None else fresh_voice()
    if st.get('stopped'):
        return voice_view(st), st, st.get('status', '')
    try:
        _commit(st, recognizer)
        st['stopped'] = True
        st['status'] = '✅ التسجيل انتهى؛ راجع Ask ?? واضغط Send.'
    except Exception as exc:
        st['status'] = f'خطأ عند إنهاء التسجيل: {type(exc).__name__}: {str(exc)[:120]}'
    return voice_view(st), st, st['status']


def transcription_of_file(audio, current_draft: str, recognizer=transcribe_signal, mode='auto'):
    if audio is None:
        return current_draft or '', 'سجل أو ارفع صوتًا أولًا.'
    report=audio_diagnostics(audio)
    if not report['ok']:
        return current_draft or '', '⚠️ '+report['reason']
    wave = normalize_audio(audio)
    if len(wave) == 0:
        return current_draft or '', 'الملف الصوتي فارغ.'
    if len(wave) > 120 * SAMPLE_RATE:
        return current_draft or '', 'ارفع تسجيلًا أقصر من دقيقتين للتجربة.'
    try:
        # Allow legacy injected single-argument recognizers in unit tests.
        result = transcribe_signal(wave, mode=mode) if recognizer is transcribe_signal else recognizer(wave)
        if not result:
            return current_draft or '', '⚠️ وصل الصوت لكن Whisper مرجعش كلمات واضحة. جرّب وضع أعلى دقة أو سجّل بجملة أطول. '+describe_audio(audio)
        return append_text(current_draft or '', result), '✅ اتفرّغ التسجيل كاملًا؛ راجع النص قبل الإرسال. '+describe_audio(audio)
    except Exception as exc:
        return current_draft or '', f'مشكلة في تحويل الصوت: {type(exc).__name__}: {str(exc)[:140]}'


def edit_distance(a, b) -> int:
    """Deterministic Levenshtein distance for a short ASR reference and hypothesis."""
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        curr=[i]
        for j,y in enumerate(b,1):
            curr.append(min(curr[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=curr
    return prev[-1]


def normalize_reference(text: str) -> str:
    import unicodedata
    t=unicodedata.normalize('NFKC',text or '').lower()
    t=re.sub(r'[\u064b-\u065f\u0670\u0640]','',t)
    t=t.translate(str.maketrans('أإآى','اااي'))
    return re.sub(r'[^\w\u0621-\u064a]+',' ',t).strip()


def speech_accuracy_test(audio, reference, mode='auto'):
    """Actually transcribe ONE captured recording on the user's Colab runtime."""
    info=audio_diagnostics(audio)
    if not info['ok']:
        return '', '⚠️ '+info['reason']
    reference=normalize_reference(reference)
    if not reference:
        return '', '⚠️ اكتب النص اللي قلته في خانة Reference عشان نحسب WER.'
    import time
    start=time.perf_counter()
    try:
        recognized=transcribe_signal(normalize_audio(audio),mode)
        secs=time.perf_counter()-start
        truth=reference.split()
        candidate=normalize_reference(recognized).split()
        wer=edit_distance(truth,candidate)/max(len(truth),1)
        ref_chars=list(reference.replace(' ',''))
        hyp_chars=list(normalize_reference(recognized).replace(' ',''))
        cer=edit_distance(ref_chars,hyp_chars)/max(len(ref_chars),1)
        return recognized,(f'📊 **اختبار صوت حقيقي على تسجيلك** — المدة {info["duration_s"]} ثانية؛ '
                           f'وقت التفريغ {secs:.1f} ثانية؛ '
                           f'**WER={wer:.1%}، CER={cer:.1%}**. '
                           'الأقل أفضل. الأرقام دي لتسجيلك ده فقط، مش دقة عامة للنموذج.')
    except Exception as exc:
        return '',f'⚠️ ASR benchmark failed: {type(exc).__name__}: {str(exc)[:180]}'
