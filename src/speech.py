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


@lru_cache(maxsize=1)
def load_asr():
    """Lazy ASR load; GPU high-accuracy; CPU lightweight alternative.

    This does not need Qwen chat and can run while Colab GPU is exhausted.
    """
    import torch
    from faster_whisper import WhisperModel
    gpu = bool(torch.cuda.is_available())
    model_pref = os.getenv('DEGHEIDY_STT_MODEL', 'large-v3' if gpu else 'small').strip()
    if gpu:
        candidates = [model_pref] + [x for x in ('medium', 'small') if x != model_pref]
        device, compute = 'cuda', 'float16'
    else:
        candidates = [model_pref] + [x for x in ('small', 'base') if x != model_pref]
        device, compute = 'cpu', 'int8'
    last_error = None
    for key in candidates:
        try:
            print(f'Loading ASR {key} on {device}/{compute} (model downloads only once if cached)')
            m = WhisperModel(key, device=device, compute_type=compute,
                             cpu_threads=4 if not gpu else 0)
            print('ASR model ready:', key)
            return m, f'Whisper {key} ({device})'
        except Exception as exc:
            last_error = exc
            print('Speech engine load failed:', key, str(exc)[:180])
            if gpu:
                torch.cuda.empty_cache()
    raise RuntimeError('All speech models failed to load: '+str(last_error))


def transcribe_signal(signal: np.ndarray) -> str:
    """No language switch; VAD + no previous text conditioning mitigates hallucination."""
    signal = np.asarray(signal, dtype=np.float32)
    if signal.size < SAMPLE_RATE * .4 or np.sqrt(np.mean(signal**2)) < .002:
        return ''
    with ASR_LOCK:
        model, _ = load_asr()
        segments, info = model.transcribe(
            signal, task='transcribe', language=None, beam_size=5,
            temperature=0, condition_on_previous_text=False,
            vad_filter=True, vad_parameters={'min_silence_duration_ms':500},
            multilingual=True,  # re-detect language across segments when possible
            no_speech_threshold=.65, compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
        )
        parts = []
        for seg in segments:
            txt = re.sub(r'\s+', ' ', seg.text).strip()
            if txt:
                parts.append(txt)
    return ' '.join(parts)


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


def transcription_of_file(audio, current_draft: str, recognizer=transcribe_signal):
    if audio is None:
        return current_draft or '', 'سجل أو ارفع صوتًا أولًا.'
    wave = normalize_audio(audio)
    if len(wave) == 0:
        return current_draft or '', 'الملف الصوتي فارغ.'
    if len(wave) > 120 * SAMPLE_RATE:
        return current_draft or '', 'ارفع تسجيلًا أقصر من دقيقتين للتجربة.'
    try:
        result = recognizer(wave)
        if not result:
            return current_draft or '', 'مفيش كلام واضح اتسمع في التسجيل.'
        return append_text(current_draft or '', result), '✅ اتفرّغ التسجيل تلقائيًا؛ راجع النص.'
    except Exception as exc:
        return current_draft or '', f'مشكلة في تحويل الصوت: {type(exc).__name__}: {str(exc)[:140]}'
