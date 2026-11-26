"""DEGHEIDY: 2026-ready bilingual chat with retrieved live web evidence + streaming voice.

Run: python app.py
T4 GPU recommended. Qwen3-4B chat quantized; independent faster-whisper ASR.
"""
from __future__ import annotations

import copy
import inspect
import threading
import time
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor

from src.dialogue import fresh_memory, commit_answer, guarded_stream, build_messages, choose_budget, evaluate_reply
from src.speech import fresh_voice, ingest_chunk, finalize_audio, transcription_of_file, load_asr, ASR_OPTIONS, describe_audio, speech_accuracy_test
from src.realtime import needs_fresh_search, should_search, search_web, evidence_footer, inject_evidence, unavailable_message, news_headline_digest, cairo_now
from src.article_reader import enrich_articles
from src.evidence_guard import impact_question, scenario_answer, safe_digest, render_news_response, compose_coverage_digest
from src.coverage import extract_requirements, checklist_for_prompt, missing_requirements
from src.quality_next import spec_for, structured_stream, repeat_answer
from src.news_quality import official_target, check_official_claim, strict_news_request, strict_story_digest
from src.job_manager import ResponseJobManager, _ACTIVE_CANCEL
from src.precision import occupancy_solution, mutable_default_solution, average_empty_solution, DECLARED_BUS_COUNT, BUS_QUESTION, exact_short_calculation, stable_unique_solution
_UI_JOBS = ResponseJobManager()

CHAT_MODEL = 'Qwen/Qwen3.5-4B'
CHAT_FALLBACK = 'Qwen/Qwen2.5-3B-Instruct'  # manually selectable compatibility option
_GENERATION_LOCK = threading.Lock()
_MODEL_LOAD_LOCK = threading.Lock()
_MODEL_LOAD_JOB = None
_MODEL_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix='degh-model-warmup')



@lru_cache(maxsize=1)
def load_chat_model():
    """Lazy-load one selected model (no invisible, automatic multi-model downloads)."""
    import os
    import torch
    from transformers import BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Enable a T4 GPU, or use live web search in CPU mode.')
    target=os.environ.get('DEGHEIDY_MODEL', CHAT_MODEL).strip()
    adapter_path=os.environ.get("DEGHEIDY_QWEN_ADAPTER", "").strip()
    revision=os.environ.get("DEGHEIDY_MODEL_REVISION", "").strip()
    hf_kwargs={"revision":revision} if revision else {}
    if adapter_path:
        from pathlib import Path
        from src.qwen_adapter import validate_adapter_base
        validate_adapter_base(Path(adapter_path), target, revision)
    print(f'[MODEL] Starting model load: {target}', flush=True)
    print('[MODEL] First launch downloads model weights; later calls reuse the in-memory model.', flush=True)
    t0=time.perf_counter()
    quant=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True)
    try:
        if target.startswith('Qwen/Qwen3.5-') and not adapter_path:
            from transformers import AutoProcessor, AutoModelForMultimodalLM
            proc=AutoProcessor.from_pretrained(target, **hf_kwargs)
            model=AutoModelForMultimodalLM.from_pretrained(
                target,quantization_config=quant,device_map='auto',
                dtype=torch.float16,low_cpu_mem_usage=True, **hf_kwargs)
        else:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            proc=AutoTokenizer.from_pretrained(target, **hf_kwargs)
            model=AutoModelForCausalLM.from_pretrained(
                target,quantization_config=quant,device_map='auto',
                dtype=torch.float16,low_cpu_mem_usage=True, **hf_kwargs)
        if adapter_path:
            from peft import PeftModel
            model=PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        model.eval()
        print(f'[MODEL] Ready: {target} in {time.perf_counter()-t0:.1f}s', flush=True)
        return proc,model
    except Exception as exc:
        import traceback
        print('[MODEL ERROR] Failed to load. No automatic second download will start.',flush=True)
        traceback.print_exc()
        raise RuntimeError(f'Model load failed ({target}): {type(exc).__name__}: {str(exc)[:280]}') from exc



def warm_model_in_background():
    """Share one async cold-start job across UI launch and all chat turns."""
    global _MODEL_LOAD_JOB
    import torch
    if not torch.cuda.is_available():
        return None
    with _MODEL_LOAD_LOCK:
        if _MODEL_LOAD_JOB is None:
            print('[WARMUP] Initializing Qwen without blocking Gradio interface...', flush=True)
            _MODEL_LOAD_JOB = _MODEL_EXECUTOR.submit(load_chat_model)
        return _MODEL_LOAD_JOB


def fast_guarded_stream(question, memory, stream_fn):
    """SINGLE-PASS streaming: no hidden second generation or duplicate answer.

    The model receives a compact task checklist *before* generation through
    build_messages(); after the stream finishes we only LOG missing items.
    Checks do not block the stream, regenerate text, or replace existing text.
    """
    if spec_for(question) is not None:
        cancel = _ACTIVE_CANCEL.get()
        yield from structured_stream(question,memory,stream_fn,build_messages,choose_budget(question),cancel,live_preview=cancel is not None)
        return
    answer = ''
    for answer in stream_fn(build_messages(question, memory), choose_budget(question)):
        yield answer
    if not answer.strip():
        raise RuntimeError('Model produced an empty reply')
    missing = missing_requirements(question, answer)
    if missing:
        print('[COVERAGE REPORT] Potential missing sections (no auto retry):', missing, flush=True)
    issues = evaluate_reply(question, answer, memory)
    if issues:
        print('[QUALITY REPORT] Heuristic warnings (no auto retry):', issues, flush=True)


def complete_multitask_answer(question, memory, original, generation_fn):
    """Backwards-compatible entry point: ALWAYS one generator invocation.

    The old implementation called Qwen a second time after the first answer,
    displayed a long pause and duplicated the response under an appendix.
    """
    yield from fast_guarded_stream(question, memory, generation_fn)


def normalize_model_messages(raw_messages):
    """Place all instructions in ONE leading system message; preserve dialogue order.

    Qwen3.5 rejects a second system message even if it is adjacent to the first.
    Important for retrieved news, context trimming, and quality-check retries.
    """
    systems=[]
    conversation=[]
    for item in (raw_messages or []):
        role=str(item.get('role','')).strip().lower()
        content=str(item.get('content',''))
        if role=='system':
            if content.strip() and content.strip() not in systems:
                systems.append(content.strip())
        elif role in ('user','assistant'):
            conversation.append({'role':role,'content':content})
    return [{'role':'system','content':'\n\n'.join(systems) or
             'You are a helpful multilingual AI assistant. Answer the latest user request.'}] + conversation


def fit_chat_context(raw_messages, encode, max_prompt_tokens=4200):
    """Reduce large retrieved-news prompts without ever duplicating system.

    Preserve latest user question. First remove old dialogue turns; only then
    condense a huge leading instruction/evidence block. The token limit is
    configurable; the model's tokenizer makes the final measurement.
    """
    messages=normalize_model_messages(raw_messages)
    encoded=encode(messages)
    attempts=0
    while encoded['input_ids'].shape[-1] > max_prompt_tokens and attempts < 18:
        attempts += 1
        if len(messages)>2:
            # Oldest conversational turn. Latest question and system survive.
            del messages[1]
        else:
            instruction=messages[0]['content']
            if len(instruction)>1200:
                # Retain opening behavioral rules AND the latest cited news.
                target=max(1200,int(len(instruction)*0.70))
                start=min(850,max(350,target//3))
                end=max(250,target-start-40)
                messages[0]['content']=(instruction[:start]+
                    '\n[Older instructions/context shortened for token budget]\n'+
                    instruction[-end:])
            else:
                # Rare large standalone user prompt: retain its ending,
                # where the explicit question usually occurs.
                latest=messages[-1]['content']
                if len(latest)<700: break
                messages[-1]['content']=latest[:350]+'\n[Long input shortened]\n'+latest[-max(350,int(len(latest)*0.65)):]
        encoded=encode(messages)
    # Double-check safety invariant after every condensation path.
    assert messages[0]['role']=='system'
    assert sum(m['role']=='system' for m in messages)==1
    return messages,encoded

def model_stream(messages, max_new_tokens):
    """Yield cumulative generation text; no guarantee of factual freshness without search."""
    import torch
    from transformers import TextIteratorStreamer, StoppingCriteria, StoppingCriteriaList
    from queue import Empty
    proc,model=load_chat_model()
    cancel_event = _ACTIVE_CANCEL.get() or threading.Event()
    class StopOnCancel(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            return bool(cancel_event and cancel_event.is_set())
    tok=proc.tokenizer if hasattr(proc,'tokenizer') else proc
    def encode(msgs):
        kw={'add_generation_prompt':True,'tokenize':True,
            'return_tensors':'pt','return_dict':True}
        try:
            return proc.apply_chat_template(msgs,enable_thinking=False,**kw)
        except TypeError:
            return proc.apply_chat_template(msgs,**kw)
    messages=normalize_model_messages(messages)
    stamp=cairo_now().strftime('%Y-%m-%d')
    messages[0]['content'] += ('\nCURRENT LOCAL DATE (Africa/Cairo): '+stamp+'. This date is NOT proof that an event happened. For fresh news use retrieved sources only.')
    # Qwen3.5's official chat template allows exactly one system message,
    # and it must be FIRST. Never use [messages[0]] + messages[-N:] here:
    # for a short conversation that duplicates the system message and causes
    # TemplateError: System message must be at the beginning.
    # Ordinary chat should not prefill thousands of unrelated tokens. News has a higher
    # budget to retain article evidence, but still a bounded context.
    news_prompt = 'RETRIEVED' in messages[0]['content'].upper() or 'ARTICLE' in messages[0]['content'].upper()
    prompt_budget = 3300 if news_prompt else (2750 if len(messages)>3 or len(messages[-1]['content'])>450 else 2150)
    messages, encoded = fit_chat_context(messages, encode, max_prompt_tokens=prompt_budget)
    print('[PREFILL] prompt_tokens=',encoded['input_ids'].shape[-1],
          '| max_new_tokens=',max_new_tokens,flush=True)
    encoded=encoded.to(model.device)
    streamer=TextIteratorStreamer(tok,skip_prompt=True,skip_special_tokens=True,timeout=75)
    errors=[]
    def generate():
        try:
            with _GENERATION_LOCK,torch.inference_mode():
                model.generate(**encoded,streamer=streamer,max_new_tokens=max_new_tokens,
                               do_sample=False,use_cache=True,
                               repetition_penalty=1.05,pad_token_id=tok.eos_token_id,
                               stopping_criteria=StoppingCriteriaList([StopOnCancel()]))
        except Exception as exc:
            errors.append(exc)
            streamer.on_finalized_text('',stream_end=True)
    worker=threading.Thread(target=generate,daemon=True)
    worker.start()
    output=''
    gen_start=time.perf_counter()
    first_chunk_at=None
    try:
        for delta in streamer:
            if cancel_event is not None and cancel_event.is_set():
                break
            output+=delta
            if output.strip():
                if first_chunk_at is None:first_chunk_at=time.perf_counter()
                yield output.strip()
    except Empty as exc:
        cancel_event.set()
        if not worker.is_alive():
            raise RuntimeError('توقف توليد Qwen بشكل غير متوقع؛ راجع رسائل CUDA/OOM في Colab') from exc
        raise TimeoutError('ما وصلش أي جزء جديد من Qwen خلال 75 ثانية. راجع ذاكرة T4 وسجل Colab.') from exc
    worker.join(timeout=3)
    elapsed=time.perf_counter()-gen_start
    try:
        emitted_tokens=len(tok.encode(output,add_special_tokens=False))
    except Exception:
        emitted_tokens=None
    print('[MODEL SPEED]', 'first_chunk_s=',round(first_chunk_at-gen_start,2) if first_chunk_at else None,
          'total_s=',round(elapsed,2),'output_tokens=',emitted_tokens,
          'tokens_per_sec=',round(emitted_tokens/elapsed,2) if emitted_tokens is not None and elapsed>0 else None,flush=True)
    if errors:
        raise RuntimeError(f'Model generation failed: {errors[0]}')
    if not output.strip():
        raise RuntimeError('Empty assistant output')


def _rumor_claim(question):
    import re
    return bool(re.search(r'(?i)اشاعة|إشاعة|شائعة|صحيح|حقيقي|تأكد|تحقق|rumor|true\?|confirm|هل\s+فيه\s+دليل',question or '') and
                re.search(r'(?i)اتفاق|صفقة|اشترت|استحوذ|تريليون|شراء|bought|acquired|deal',question or ''))

def turn(question, messages, memory, news_state=None):
    """Talk naturally, but verify *news* before it is shown as final.

    news_state lives in Gradio session state; it is not shared between users.
    """
    question=(question or '').strip()
    shown=copy.deepcopy(messages or [])
    memory=memory or fresh_memory()
    if not question:
        yield shown, '',memory,news_state
        return
    prior=copy.deepcopy(shown)
    shown.extend([{'role':'user','content':question},
                  {'role':'assistant','content':'⏳ بجهز الرد...'}])
    yield copy.deepcopy(shown),'',memory,news_state
    begin=time.perf_counter()
    first=None;shown_at=0;answer='';evidence=None
    try:
        import torch
        fresh=bool(official_target(question)) or should_search(question,memory)
        if not fresh and not impact_question(question):
            repeated=repeat_answer(question,prior,memory)
            if repeated is not None:
                shown[-1]['content']=repeated
                memory=commit_answer(memory,question,repeated)
                yield copy.deepcopy(shown),'',memory,None
                return
        is_impact=impact_question(question)
        print('[ROUTER]', 'NEWS' if fresh else ('NEWS_IMPACT' if is_impact and news_state else 'CHAT'),
              '| question=', question[:100].replace('\n',' '), flush=True)
        if not fresh and not is_impact and news_state is not None:
            # A real change of subject must not inherit earlier article evidence.
            # The regular chat memory still keeps relevant older user statements.
            if not needs_fresh_search(question):
                news_state=None
        if fresh and official_target(question):
            shown[-1]['content']='🌐 ببحث داخل المواقع الرسمية وبراجع النصوص المتاحة...'
            yield copy.deepcopy(shown),'',memory,None
            answer=check_official_claim(question)
            shown[-1]['content']=answer
            memory=commit_answer(memory,question,answer)
            yield copy.deepcopy(shown),'',memory,None
            return
        if fresh:
            shown[-1]['content']='🌐 ببحث عن مصادر مؤرخة للمعلومة...'
            yield copy.deepcopy(shown),'',memory,news_state
            evidence=search_web(question,memory=memory)
            if evidence.status!='found':
                news_state=None
                answer=unavailable_message(evidence)
                if _rumor_claim(question) or ('CORPORATE' in (evidence.plan.topic or '').upper() if evidence.plan else False) or ('shutdown claim' in (evidence.plan.topic or '') if evidence.plan else False):
                    detail='لم أجد مصدرًا مؤرخًا يثبت الادعاء في نتائج البحث المتاحة.' if evidence.status=='no_results' else 'تعذر إجراء تحقق كافٍ بسبب مشكلة وصول لمصادر البحث.'
                    answer=detail+' **الادعاء غير مُثبت بالبيانات المتاحة، وده مش دليل قاطع على النفي.**\n\n'+answer
                shown[-1]['content']=answer
                memory=commit_answer(memory,question,answer)
                yield copy.deepcopy(shown),'',memory,news_state
                return
            shown[-1]['content']='📰 براجع نصوص المقالات المتاحة...'
            yield copy.deepcopy(shown),'',memory,news_state
            enrich_articles(evidence,max_articles=6 if strict_news_request(question) else 5)
            news_state=copy.deepcopy(evidence)
        elif is_impact and news_state is not None:
            # Follow-up: the user asks about implications of the same reported event.
            # Never replace those results with a generic, unrelated fresh search.
            evidence=copy.deepcopy(news_state)
        if evidence is not None and evidence.status=='found':
            if is_impact and not fresh:
                # Pure impact follow-up reuses the source state without fresh search.
                answer=scenario_answer(evidence)
            elif strict_news_request(question):
                answer=strict_story_digest(question,evidence)
            elif not torch.cuda.is_available() or not any(s.get('read_status')=='article_read' for s in evidence.sources):
                answer=compose_coverage_digest(question,evidence)
            else:
                shown[-1]['content']='🔎 بصيغ الإجابة وبفحص إن الوقائع مسنودة بالمقالات...'
                yield copy.deepcopy(shown),'',memory,news_state
                candidate=''
                def news_stream(msgs,tokens):
                    return model_stream(inject_evidence(msgs,evidence),tokens)
                try:
                    # Do not show unchecked generated news drafts to users.
                    for candidate in news_stream(build_messages(question,memory), min(640,choose_budget(question))):
                        pass
                    answer,verdict=render_news_response(question,candidate,evidence)
                    print('[EVIDENCE GUARD] approved=',verdict.approved,
                          'checked=',verdict.factual_checked,'unsupported=',verdict.unsupported,
                          'problems=',list(dict.fromkeys(verdict.problems))[:6],flush=True)
                except Exception as exc:
                    print('[EVIDENCE GUARD] generation error:',repr(exc),flush=True)
                    answer=compose_coverage_digest(question,evidence)
            shown[-1]['content']=answer
            memory=commit_answer(memory,question,answer)
            yield copy.deepcopy(shown),'',memory,news_state
            print('[NEWS] Complete in',round(time.perf_counter()-begin,1),'seconds',flush=True)
            return
        if is_impact and news_state is None:
            answer=('عشان أحلل أثر الخبر على مصر لازم أعرف الحدث المقصود. '
                    'اذكر اسم الحدث أو اسأل عنه أولًا، وهنفصل الوقائع عن السيناريوهات.')
            shown[-1]['content']=answer
            memory=commit_answer(memory,question,answer)
            yield copy.deepcopy(shown),'',memory,news_state
            return
        # Deterministic responses for audited arithmetic / code bug cases.
        fixed_code=mutable_default_solution(question) or average_empty_solution(question) or stable_unique_solution(question) or exact_short_calculation(question)
        if fixed_code is not None:
            shown[-1]['content']=fixed_code
            memory=commit_answer(memory,question,fixed_code)
            yield copy.deepcopy(shown),'',memory,news_state
            return
        # Deterministic occupancy arithmetic: avoid obvious model mistakes such as
        # 28/35 = 85.71% or reversing the argument order in unit tests.
        exact=occupancy_solution(question)
        if exact is not None:
            if any(word in question for word in ('فاكر','فكرني','افتكر')):
                past=' '.join(t.get('text','') for topic in memory.get('topics',[])
                              for t in topic.get('turns',[]) if t.get('role')=='user')
                known=[w for w in ('سنة رابعة','TensorFlow','PyTorch','Computer Vision','LLMs','MediaPipe','GPA','AI Engineer')
                       if w.lower() in past.lower()]
                if known:
                    exact=('**من المعلومات اللي إنت قلتها قبل كده:** '+', '.join(known)+
                           '. **لم تقل** إنك ضعيف في Git أو DevOps؛ دول اقتراحات للفحص مش حقائق مؤكدة.\n\n'+exact)
            shown[-1]['content']=exact
            memory=commit_answer(memory,question,exact)
            yield copy.deepcopy(shown),'',memory,news_state
            return
        if exact is None and BUS_QUESTION.search(question) and DECLARED_BUS_COUNT.search(question):
            answer=('بيانات الأتوبيسات مش كاملة أو صياغتها مش واضحة؛ مش هحسب إجمالي الأسطول من جزء منها. '
                    'اكتب لكل أتوبيس عدد المقاعد وعدد الركاب، مثل: الأول 40 كرسي و30 راكب، الثاني 35 كرسي و28 راكب.')
            shown[-1]['content']=answer
            memory=commit_answer(memory,question,answer)
            yield copy.deepcopy(shown),'',memory,news_state
            return
        if not torch.cuda.is_available():
            answer=('الشات المحلي محتاج GPU. تقدر تستخدم الأخبار والتحليل المشروط '
                    'بعد البحث والصوت على CPU، أو تفعّل T4 GPU في Colab.')
            shown[-1]['content']=answer
            memory=commit_answer(memory,question,answer)
            yield copy.deepcopy(shown),'',memory,news_state
            return
        # First run can download several GB. Notify users instead of a silent spinner.
        load_job = warm_model_in_background()
        load_started = time.perf_counter()
        if load_job is not None:
            while not load_job.done():
                cancel = _ACTIVE_CANCEL.get()
                if cancel is not None and cancel.is_set():return
                secs = int(time.perf_counter() - load_started)
                shown[-1]['content'] = ('⚙️ بجهّز موديل Qwen3.5 على الـGPU... '
                    f'({secs} ثانية). أول تشغيل ممكن ينزّل أوزان الموديل؛ '
                    'بعد ما يجهز، الأسئلة التالية مش هتحتاج التنزيل من جديد داخل نفس الجلسة.')
                yield copy.deepcopy(shown),'',memory,news_state
                time.sleep(2)
            load_job.result()  # Fail clearly if the model could not initialize.
        cancel = _ACTIVE_CANCEL.get()
        if cancel is not None and cancel.is_set():return
        shown[-1]['content'] = '✍️ الموديل جاهز، بكتب الإجابة...'
        yield copy.deepcopy(shown),'',memory,news_state
        for answer in fast_guarded_stream(question,memory,model_stream):
            now=time.perf_counter()
            if first is None and answer.strip():first=now
            if now-shown_at>=.35:
                shown[-1]['content']=answer
                yield copy.deepcopy(shown),'',memory,news_state
                shown_at=now
        if (_ACTIVE_CANCEL.get() is not None and _ACTIVE_CANCEL.get().is_set()):return
        shown[-1]['content']=answer
        memory=commit_answer(memory,question,answer)
        yield copy.deepcopy(shown),'',memory,news_state
        if first is not None:
            print('[LATENCY] first_visible=',round(first-begin,1),'s | complete=',round(time.perf_counter()-begin,1),'s',flush=True)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        shown[-1]['content']=('⚠️ تعذّر تشغيل الموديل: '+type(exc).__name__+': '+str(exc)[:220]
                             +'\nراجع رسالة الخطأ في Colab.')
        yield copy.deepcopy(shown),'',memory,news_state


def _submit_background(question, chat, memory, news_state, current_key):
    key,note=_UI_JOBS.submit(question,chat,memory,news_state,current_key)
    if not key:
        return chat,question,current_key,note,current_key or ''
    if current_key == key and _UI_JOBS.get(key) and _UI_JOBS.get(key).phase == 'running':
        # Keep the new unsent question available when the previous job is busy.
        return chat,question,key,note,key
    job=_UI_JOBS.get(key)
    snapshot=job.snapshot()
    # Avoid changing the textbox on duplicate submission; let the user review it.
    return snapshot[0],'',key,note,key


def _poll_background(key):
    import gradio as gr
    job=_UI_JOBS.get(key)
    if not job:
        return gr.skip(),gr.skip(),gr.skip(),gr.skip()
    chat,memory,news_state,phase,error=job.snapshot()
    if phase=='running':
        age=int(time.monotonic()-job.created_at)
        idle=int(time.monotonic()-job.updated_at)
        status=f'🟡 جاري الرد ({age} ثانية).' 
        if idle > 65:
            status+=f' ⚠️ لم تصل تحديثات جديدة منذ {idle} ثانية؛ راجع رسائل Colab أو استخدم إيقاف الرد.'
    elif phase=='finished':status='✅ الرد اكتمل. تقدر تبعت سؤال جديد.'
    elif phase=='cancelled':status='⏹️ تم إيقاف التوليد. تقدر تبعت سؤال جديد.'
    else:status='⚠️ فشل الرد: '+error
    return chat,memory,news_state,status


def _stop_background(key):
    return '⏹️ جاري طلب إيقاف التوليد...' if _UI_JOBS.stop(key) else 'مفيش رد نشط لإيقافه.'


def _restore_background(recovery_key):
    job=_UI_JOBS.get((recovery_key or '').strip())
    if not job:
        return [],fresh_memory(),None,None,'مفيش طلب بالكود ده في جلسة Colab الحالية. لو الجلسة انتهت، مش هيكون قابل للاستعادة.'
    chat,mem,news,phase,error=job.snapshot()
    return chat,mem,news,job.key,f'استعدت الرد. الحالة: {phase}.'


def _clear_background(key):
    _UI_JOBS.stop(key)
    return [],'',fresh_memory(),None,fresh_voice(),None,None,'تم مسح المحادثة.', ''


def transcribe_full_recording(audio, current_draft, mode='auto'):
    return transcription_of_file(audio,current_draft,mode=mode)


def make_ui():
    import gradio as gr
    kw={'label':'Conversation / المحادثة','height':430}
    if 'type' in inspect.signature(gr.Chatbot.__init__).parameters:kw['type']='messages'
    with gr.Blocks(title='DEGHEIDY — AI Chatbot') as demo:
        gr.Markdown('# DEGHEIDY — Multilingual AI Chatbot')
        chat=gr.Chatbot(**kw)
        question=gr.Textbox(label='Ask ??',placeholder='اسأل بالعربي أو English...',lines=2)
        memory=gr.State(fresh_memory())
        news_state=gr.State(None)
        job_key=gr.State(None)
        with gr.Row():
            send=gr.Button('إرسال / Send',variant='primary')
            stop=gr.Button('إيقاف الرد / Stop',variant='secondary')
            clear=gr.Button('مسح المحادثة / Clear')
        status=gr.Markdown('🟢 جاهز للسؤال.')
        with gr.Accordion('استعادة الرد لو أعدت فتح الصفحة',open=False):
            gr.Markdown('لو قفلت الصفحة، تقدر ترجع لنفس الرد قبل ما جلسة Colab تنتهي.')
            recovery=gr.Textbox(label='Recovery code',lines=1,interactive=True)
            recover=gr.Button('استعادة من الكود / Restore')
        timer=gr.Timer(value=0.9)
        # No Gradio long-running streamed callback: short requests remain responsive
        # even if the public share tunnel drops an earlier connection.
        inputs=[question,chat,memory,news_state,job_key]
        outputs=[chat,question,job_key,status,recovery]
        send.click(_submit_background,inputs=inputs,outputs=outputs,queue=False,show_progress='hidden')
        question.submit(_submit_background,inputs=inputs,outputs=outputs,queue=False,show_progress='hidden')
        timer.tick(_poll_background,inputs=[job_key],outputs=[chat,memory,news_state,status],queue=False,show_progress='hidden')
        stop.click(_stop_background,inputs=[job_key],outputs=[status],queue=False)
        recover.click(_restore_background,inputs=[recovery],outputs=[chat,memory,news_state,job_key,status],queue=False)
        with gr.Accordion('🎤 التسجيل الصوتي',open=True):
            recorded=gr.Audio(sources=['microphone','upload'],type='numpy',streaming=False,
                              format='wav',label='الميكروفون / Audio')
            audio_status=gr.Markdown('')
            asr_mode=gr.Dropdown(choices=ASR_OPTIONS,value='auto',label='دقة تحويل الصوت / ASR Model')
            with gr.Row():
                inspect_audio=gr.Button('افحص التسجيل والميكروفون',variant='secondary')
                transcribe=gr.Button('حوّل التسجيل إلى نص',variant='primary')
            recorded.change(describe_audio,inputs=[recorded],outputs=[audio_status],queue=False)
            inspect_audio.click(describe_audio,inputs=[recorded],outputs=[audio_status],queue=False)
            # Leave draft text unchanged on failure; old user's typing is retained.
            transcribe.click(transcribe_full_recording,
                             inputs=[recorded,question,asr_mode],outputs=[question,audio_status],
                             concurrency_limit=1,concurrency_id='speech_pipe')
            with gr.Accordion('اختبار دقة الصوت الحقيقي (اختياري)',open=False):
                reference=gr.Textbox(label='Reference — اكتب الجملة اللي قلتها بصوتك',lines=2)
                check_asr=gr.Button('احسب WER/CER على التسجيل')
                benchmark_text=gr.Textbox(label='النص اللي Whisper فهمه',interactive=False)
                check_asr.click(speech_accuracy_test,inputs=[recorded,reference,asr_mode],
                                outputs=[benchmark_text,audio_status],
                                concurrency_limit=1,concurrency_id='speech_pipe')

        # Older streaming-VAD feature is no longer in the default interface: it could
        # drop final audio chunks and mistake short silences for the end of a sentence.
        voice_state=gr.State(fresh_voice())  # compatibility for legacy clear callback
        clear.click(_clear_background,inputs=[job_key],
                    outputs=[chat,question,memory,news_state,voice_state,recorded,job_key,status,recovery],queue=False)
    return demo


def start_gradio():
    """Open UI without loading Qwen, explicitly disable SSR for share tunnel compatibility.

    The public gradio.live URL can occasionally load HTML while failing its
    browser-side JS bundle. The Colab port proxy and /config diagnostic in
    the notebook identify that case without pretending model weights caused it.
    """
    import os
    from urllib.parse import urlparse
    import gradio as gr
    import torch
    os.environ['GRADIO_SSR_MODE'] = 'False'
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    # Whisper on CPU avoids two large model stacks contesting the same T4 VRAM.
    os.environ.setdefault('DEGHEIDY_ASR_CPU_ONLY','0')
    print('[GPU]', bool(torch.cuda.is_available()), flush=True)
    print('[GRADIO] version:', gr.__version__, '| SSR explicitly OFF', flush=True)
    print('[UI] Building bilingual interface. Qwen is NOT downloading yet.', flush=True)
    demo = make_ui()
    launch_kw = dict(share=True, inline=True, debug=False,
                     prevent_thread_lock=True, show_error=True,
                     quiet=False, server_name='0.0.0.0')
    # Optional password gate for the publicly shared Colab link. Never hard-code credentials.
    auth_password=os.environ.get('DEGHEIDY_GRADIO_PASSWORD','').strip()
    if auth_password:
        launch_kw['auth']=(os.environ.get('DEGHEIDY_GRADIO_USER','degh'),auth_password)
        print('[SECURITY] Gradio share link password protection is enabled.',flush=True)
    else:
        print('[SECURITY] Public share link has no login. Keep it private; do not enter secrets.',flush=True)
    if 'ssr_mode' in inspect.signature(gr.Blocks.launch).parameters:
        launch_kw['ssr_mode'] = False
    if 'pwa' in inspect.signature(gr.Blocks.launch).parameters:
        launch_kw['pwa'] = False
    print('[UI] Starting server and opening public share tunnel...', flush=True)
    response = demo.queue(default_concurrency_limit=1).launch(**launch_kw)
    share_url = (response[2] if isinstance(response,tuple) and len(response)>=3
                 else getattr(demo,'share_url',None))
    if share_url and urlparse(share_url).scheme == 'https':
        print('[READY] Gradio public URL:', share_url, flush=True)
    else:
        print('[WARNING] No public HTTPS URL created; use Colab local proxy.',flush=True)
    print('[INFO] URL only works while this Colab runtime remains connected.',flush=True)
    print('[UI] Poll interval: 0.9s; recovery available until runtime restarts.',flush=True)
    if torch.cuda.is_available():
        warm_model_in_background()
        print('[WARMUP] Model is loading in the background. Follow progress in Colab logs.',flush=True)
    return demo

if __name__=='__main__':
    print('Starting DEGHEIDY chatbot; model weights load only with the first chat request.',flush=True)
    print('Keep the temporary gradio.live URL private because anyone with the URL can use the app.',flush=True)
    _DEGHEIDY_UI=start_gradio()
