"""DEGHEIDY V10: bilingual chat + no-replay/Voice Activity Detection dictation.

Run: python app.py
T4 GPU recommended. Qwen3-4B chat quantized; independent faster-whisper ASR.
"""
from __future__ import annotations

import copy
import inspect
import threading
import time
from functools import lru_cache

from src.dialogue import fresh_memory, commit_answer, guarded_stream
from src.speech import fresh_voice, ingest_chunk, finalize_audio, transcription_of_file, load_asr

CHAT_MODEL = 'Qwen/Qwen3-4B-Instruct-2507'
CHAT_FALLBACK = 'Qwen/Qwen2.5-3B-Instruct'
_GENERATION_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def load_chat_model():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('No CUDA GPU. In Colab: Runtime > Change runtime type > T4 GPU.')
    try:
        print('Loading quantized chat model:', CHAT_MODEL)
        t0=time.perf_counter()
        tok=AutoTokenizer.from_pretrained(CHAT_MODEL)
        quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16,
                                  bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True)
        model=AutoModelForCausalLM.from_pretrained(
            CHAT_MODEL, quantization_config=quant, device_map={'':0},
            dtype=torch.float16, attn_implementation='sdpa')
        model.eval()
        print(f'Qwen3-4B ready ({time.perf_counter()-t0:.1f}s).')
        return tok,model
    except Exception as exc:
        print('4B load failed, using proven Qwen2.5-3B fallback:', repr(exc)[:400])
        import gc
        gc.collect();torch.cuda.empty_cache()
        tok=AutoTokenizer.from_pretrained(CHAT_FALLBACK)
        model=AutoModelForCausalLM.from_pretrained(
            CHAT_FALLBACK,dtype=torch.float16,attn_implementation='sdpa',low_cpu_mem_usage=True).to('cuda')
        model.eval()
        return tok,model


def model_stream(messages, max_new_tokens):
    """Yield *cumulative* assistant text for Gradio streaming."""
    import torch
    from transformers import TextIteratorStreamer
    tok,model=load_chat_model()
    encoded=tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_tensors='pt')
    if encoded.shape[-1]>2600:
        messages=[messages[0]]+messages[-5:]
        encoded=tok.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_tensors='pt')
    encoded=encoded.to('cuda')
    streamer=TextIteratorStreamer(tok,skip_prompt=True,skip_special_tokens=True,timeout=120)
    errors=[]
    def generate():
        try:
            with _GENERATION_LOCK,torch.inference_mode():
                model.generate(input_ids=encoded,streamer=streamer,max_new_tokens=max_new_tokens,
                               do_sample=False,use_cache=True,
                               repetition_penalty=1.06,pad_token_id=tok.eos_token_id)
        except Exception as exc:
            errors.append(exc)
            streamer.on_finalized_text('',stream_end=True)
    worker=threading.Thread(target=generate,daemon=True)
    worker.start()
    output=''
    for delta in streamer:
        output+=delta
        if output.strip(): yield output.strip()
    worker.join(timeout=4)
    if errors: raise RuntimeError(f'Model generation failed: {errors[0]}')
    if not output.strip():raise RuntimeError('Empty assistant output')


def turn(question, messages, memory):
    question=(question or '').strip()
    shown=copy.deepcopy(messages or [])
    memory=memory or fresh_memory()
    if not question:
        yield shown, '',memory
        return
    shown.extend([{'role':'user','content':question},{'role':'assistant','content':'...'}])
    yield copy.deepcopy(shown),'',memory
    begin=time.perf_counter()
    first=None;shown_at=0; answer=''
    try:
        for answer in guarded_stream(question,memory,model_stream):
            now=time.perf_counter()
            if first is None and answer.strip():first=now
            if now-shown_at >= .15:
                shown[-1]['content']='⏳ **مسودة جارية (قد تتغير بعد مراجعة الجودة):**\n\n'+answer
                yield copy.deepcopy(shown),'',memory
                shown_at=now
        shown[-1]['content']=answer
        memory=commit_answer(memory,question,answer)
        yield copy.deepcopy(shown),'',memory
        if first is not None:
            print(f'First token visible: {first-begin:.1f}s | Response complete: {time.perf_counter()-begin:.1f}s')
    except Exception:
        import traceback
        traceback.print_exc()
        shown[-1]['content']='حصل خطأ في تشغيل الموديل. شوف تفاصيل الخطأ تحت في Colab.'
        yield copy.deepcopy(shown),'',memory


def make_ui():
    import gradio as gr
    import torch
    gpu = bool(torch.cuda.is_available())
    kw={'label':'Conversation / المحادثة','height':430}
    if 'type' in inspect.signature(gr.Chatbot.__init__).parameters:kw['type']='messages'
    with gr.Blocks(title='DEGHEIDY Multilingual AI V10') as demo:
        gr.Markdown('# DEGHEIDY — Multilingual AI Chatbot V10')
        if not gpu:
            gr.Markdown('**CPU mode:** الإملاء الصوتي شغال بموديل خفيف، لكن Chat AI محتاج GPU. احتفظ بالنص من Ask ?? لحد ما الـGPU يتاح.')
        chat=gr.Chatbot(**kw)
        question=gr.Textbox(label='Ask ??',placeholder='',lines=2)
        memory=gr.State(fresh_memory())
        voice_state=gr.State(fresh_voice())
        with gr.Row():
            send=gr.Button('إرسال / Send',variant='primary',interactive=gpu)
            clear=gr.Button('مسح المحادثة / Clear')
        with gr.Accordion('🎙️ كلام مباشر → نص | عربي + English تلقائيًا',open=True):
            gr.Markdown('**مفيش اختيار لغة.** سجّل واتكلم عربي وإنجليزي. اللي هيتكتب أثناء الكلام مؤقت وممكن يتعدل؛ بعد وقفة قصيرة النص بيتثبت من غير إعادة تفريغ الصوت القديم. اضغط إيقاف وراجع Ask ??.')
            mic=gr.Audio(sources=['microphone'],type='numpy',streaming=True,label='🎙️ Live voice — auto Arabic + English (no language switch)')
            voice_status=gr.Markdown('Automatic Arabic/English. VAD stops text appearing during silence; ASR model is loaded on first recording.')
        with gr.Accordion('🎧 بديل: تسجيل أو ملف صوتي كامل',open=False):
            recorded=gr.Audio(sources=['microphone','upload'],type='numpy',label='Audio recording / ملف صوت')
            transcribe=gr.Button('تحويل الصوت إلى نص / Transcribe')
            file_status=gr.Markdown('هتراجع النص قبل إرساله للشات.')
        send.click(turn,inputs=[question,chat,memory],outputs=[chat,question,memory],show_progress='hidden')
        question.submit(turn,inputs=[question,chat,memory],outputs=[chat,question,memory],show_progress='hidden')
        mic.start_recording(fresh_voice,inputs=[question],outputs=[voice_state],queue=False)
        mic.stream(ingest_chunk,inputs=[mic,voice_state],outputs=[question,voice_state,voice_status],
                   stream_every=1.8,time_limit=120,concurrency_limit=1,concurrency_id='speech_pipe',trigger_mode='multiple')
        mic.stop_recording(finalize_audio,inputs=[voice_state],outputs=[question,voice_state,voice_status],
                           concurrency_limit=1,concurrency_id='speech_pipe')
        transcribe.click(transcription_of_file,inputs=[recorded,question],outputs=[question,file_status],concurrency_limit=1,concurrency_id='speech_pipe')
        clear.click(lambda:([], '',fresh_memory(), fresh_voice(),None),
                    outputs=[chat,question,memory,voice_state,mic],queue=False)
    return demo


if __name__=='__main__':
    import torch
    if torch.cuda.is_available():
        load_chat_model()
        print('Qwen3 4B chat ready. Speech model downloads lazily the first time you record.')
    else:
        print('No Colab GPU: voice-only CPU mode with a smaller Whisper model. Send is disabled.')
    print('Open the HTTPS Gradio share link in a NEW tab to allow microphone access. Keep Colab running.')
    print('Voice audio passes through your Colab/Gradio temporary server. Keep the share link private.')
    make_ui().queue(default_concurrency_limit=1).launch(share=True,inline=True,debug=True,show_error=True)
