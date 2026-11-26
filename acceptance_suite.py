"""Actual 10-case acceptance runner for the user's Google Colab T4 session.

No synthetic Qwen answers. Test 9 needs an optional USER-recorded WAV; test 10
actually exercises the background job's Stop path. Never auto-claim 8/10.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import re
import time
import os,uuid,hashlib,importlib.metadata
from typing import Optional

from src.coverage import missing_requirements
from src.dialogue import fresh_memory

CASES = [
  ('01_career', "I'm a fourth-year AI student وعندي مشاريع في TensorFlow وComputer Vision وLLMs. عايز ألاقي شغل AI Engineer مدفوع، ومش عايز Internships ولا كورسات مبتدئين. اعملي خطة 3 شهور، لكل شهر 3 مهام عملية ونتيجة قابلة للقياس. وبعدها حدد أول وظيفتين مناسبين لمستوايا، من غير ما تفترض خبرة Production عندي."),
  ('02_edit', 'أنا كمان عندي Gym Exercise Analysis باستخدام MediaPipe وOpenCV، ومشروع Multilingual Chatbot. خبرتي في Deployment محدودة. عدّل الشهر الثاني فقط من الخطة اللي فاتت إلى 3 مهام مناسبة لمشاريعي. متعيدش باقي الخطة.'),
  ('03_buses', 'عندي 3 أتوبيسات: الأول سعته 36 كرسي وراكب فيه 27، والثاني سعته 30 كرسي وفيه 24 راكب، والثالث سعته 20 كرسي وفيه 11 راكب. احسب إشغال الأول والثاني والثالث، والإجمالي للأسطول، ورتبهم من الأقل إشغالًا للأعلى. راجع كل الأرقام قبل ما تجاوب.'),
  ('04_code', 'راجع الكود ده:\n\ndef save_result(value, results=[]):\n    results.append(value)\n    return results\n\nاشرح العيب، وصحح الدالة، واكتب 3 اختبارات pytest: استدعاءان مستقلان، تمرير قائمة موجودة، وإضافة قيمة سالبة. ممنوع تقول إنك نفذت الاختبارات إلا لو شغلتها فعلًا.'),
  ('05_memory', 'نرجع لموضوع شغلي. لخص كل المهارات والمشاريع اللي أنا ذكرتها في المحادثة دي، وقولي أنا صرحت إن خبرتي في إيه محدودة. وبعدها رشح أولويتين للتطوير. افصل بين معلوماتي المؤكدة واقتراحاتك، في أقل من 130 كلمة.'),
  ('06_rag', 'عندي Chatbot بيقرأ ملفات PDF بتتغير يوميًا، ولازم يفتكر آخر 5 رسائل. اشرح الفرق بين RAG وFine-tuning وConversation Memory بالمصري، وصمم Architecture من 5 خطوات تبدأ بتحديث الملفات وتنتهي برد مدعوم بمصدر. بعد كده اذكر 3 أخطاء تقنية لازم أتجنبها. متسيبش أي جزء ومتكررّش.'),
  ('07_news', 'ابحث عن أهم خبرين سياسيين دوليين حصل فيهما تطور خلال آخر 12 ساعة بتوقيت القاهرة. لكل خبر اذكر الحدث، وتاريخ وقوعه إن عُرف، وتاريخ نشر التقرير، وملخصًا واقعيًا، ورابطًا مباشرًا للمصدر الأصلي. لا تضع مقالات رأي أو توقعات فلكية ضمن الأخبار، ولا تخترع تفاصيل لو المقال غير متاح.'),
  ('08_rumor', 'قرأت منشورًا بيقول إن شركة OpenAI أعلنت إيقاف ChatGPT بالكامل الأسبوع القادم. هل فيه إعلان رسمي يؤكد ده؟ ابحث عن دليل، وحدد إيه اللي تقدر تثبته وإيه اللي مش متحقق منه. ومتقولش إنك راجعت بيانًا رسميًا إلا لو عندك مصدره.'),
]
AUDIO_REFERENCE='أنا يوسف، and I am working on a Computer Vision project using MediaPipe and PyTorch، وعايز deploy my AI chatbot with FastAPI and Docker.'
STOP_PROMPT='اشرح بالتفصيل ازاي أبني نظام MLOps كامل فيه Data Pipeline وModel Training وDeployment وMonitoring مع أمثلة Python.'
AFTER_STOP='احسب 19 × 23 واكتب الرقم فقط'


def _flags(case,answer):
    """Specific objective signals. A true flag is NOT a validated natural-language grade."""
    a=answer.lower()
    if case=='03_buses':
        return {v:v in answer for v in ['75%','80%','55%','72.09%','62 ÷ 86','الثالث (55%) ← الأول (75%) ← الثاني (80%)']}
    if case=='04_code':
        return {'new_none_default':bool(re.search(r'(?m)def\s+save_result\s*\([^\n]*results\s*=\s*None',answer)),
                'three_tests':len(re.findall(r'(?m)^\s*def\s+test_',answer))>=3,
                'keeps_existing_list':bool(re.search(r'(?i)existing|قائمة\s+موجودة|\[10,\s*20\]',answer))}
    if case=='05_memory':
        return {x:x.lower() in a for x in ['TensorFlow','Computer Vision','LLMs','MediaPipe','OpenCV','Multilingual Chatbot','Deployment']}
    if case=='06_rag':
        return {'mentions_retrieval':bool(re.search(r'(?i)retriev|استرجاع|بحث\s+دلالي',answer)),
                'mentions_vector_index':bool(re.search(r'(?i)vector|الفهرس|الفهرسة|قاعدة\s+متجهة',answer))}
    if case=='07_news':
        return {'direct_source_urls':bool(re.search(r'https?://',answer)),
                'no_opinion_as_story':not bool(re.search(r'(?im)^\s*[12][.)]\s*.*(?:رأي\s*:|opinion\s*:|astrological)',answer))}
    if case=='08_rumor':
        return {'not_invented_press_release':'أثبتت التحقيقات' not in a,
                'uncertainty_stated':bool(re.search(r'(?i)غير\s+مثبت|لم\s+أجد|مافيش\s+دليل|لا\s+أستطيع\s+تأكيد|غير\s+مؤكد',answer))}
    return {}


def _record(name, question, answer, seconds=None, error=None):
    flags=_flags(name,answer)
    return {
      'case':name,'question':question,'answer':answer,
      'seconds':round(seconds,2) if seconds is not None else None,
      'runtime_error':error,'human_score':None,'critical_error':None,
      'mechanical_flags':flags,
      'missing_heuristics':missing_requirements(question,answer) if answer else ['empty'],
      'important_note':'Objective flags and missing heuristics are NOT a human accuracy score.'
    }


def _hash_files(paths, root):
    h=hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(root)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def run_metadata():
    import torch
    root=Path(__file__).resolve().parent
    adapter=os.getenv('DEGHEIDY_QWEN_ADAPTER','')
    versions={}
    for pkg in ('torch','transformers','bitsandbytes','peft','gradio','faster-whisper'):
        try:versions[pkg]=importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:versions[pkg]='NOT_INSTALLED'
    return {'run_type':'REAL_T4_APP','run_id':str(uuid.uuid4()),
        'code_sha256':_hash_files([root/'app.py',*list((root/'src').glob('*.py'))],root),
        'suite_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_model':os.getenv('DEGHEIDY_MODEL','Qwen/Qwen3.5-4B'),
        'model_revision':os.getenv('DEGHEIDY_MODEL_REVISION','UNPINNED'),
        'adapter_sha256':_hash_files([p for p in Path(adapter).glob('*') if p.suffix in ('.json','.safetensors','.bin')],Path(adapter)) if adapter else 'BASE_NO_ADAPTER',
        'runtime_versions':versions,'gpu':torch.cuda.get_device_name(0)}


def _save(path,records,metadata):
    Path(path).write_text(json.dumps({**metadata,
       'generated_at':datetime.now().isoformat(),
       'certification':'NOT CERTIFIED 8/10; review all real outputs',
       'audio_case_requires_real_user_recording':True,
       'records':records},ensure_ascii=False,indent=2),encoding='utf-8')


def run_ten_case_acceptance(output_path='/content/DEGHEIDY_Real_10_Case_Results.json',audio_path:Optional[str]=None):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('لا يوجد CUDA/T4: لم يتم سؤال Qwen الحقيقي. شغّل الملف في Colab مع GPU.')
    import app
    from src.speech import speech_accuracy_test
    from src.job_manager import ResponseJobManager
    from src.dialogue import choose_budget
    metadata=run_metadata()
    memory=fresh_memory(); messages=[]; news_state=None; records=[]
    for idx,(name,prompt) in enumerate(CASES,1):
        t=time.perf_counter();last=None
        print(f'[{idx}/10] Actual app.turn {name}, budget={choose_budget(prompt)}',flush=True)
        try:
            for snapshot in app.turn(prompt,messages,memory,news_state):
                last=snapshot
            if last is None:raise RuntimeError('No frames')
            messages,_,memory,news_state=last
            answer=messages[-1]['content']
            error=answer[:80] if answer.startswith('⚠️') else None
        except Exception as exc:
            answer='';error=f'{type(exc).__name__}: {str(exc)[:150]}'
        rec=_record(name,prompt,answer,time.perf_counter()-t,error)
        records.append(rec);_save(output_path,records,metadata)
        print('=>',name,'sec:',rec['seconds'],'chars:',len(answer),'missing:',rec['missing_heuristics'],flush=True)
    if audio_path:
        # scipy comes with Colab and can read WAV recorded by the user.
        try:
            from scipy.io import wavfile
            sr,raw=wavfile.read(audio_path)
            t=time.perf_counter()
            hyp,status=speech_accuracy_test((sr,raw),AUDIO_REFERENCE)
            rec=_record('09_voice',AUDIO_REFERENCE,hyp,time.perf_counter()-t)
            rec['asr_diagnostics']=status
        except Exception as exc:
            rec=_record('09_voice',AUDIO_REFERENCE,'',error=repr(exc))
    else:
        rec=_record('09_voice',AUDIO_REFERENCE,'',error='NOT TESTED: provide an actual user recording and compare WER/CER')
    records.append(rec);_save(output_path,records,metadata)

    # Stop test uses the REAL application worker and real Qwen weights.
    t=time.perf_counter(); manager=ResponseJobManager(); answer='';error=None
    try:
        key,note=manager.submit(STOP_PROMPT,messages,memory,news_state)
        if not key:raise RuntimeError(note)
        # Wait for the job to start; stop as soon as it starts working.
        time.sleep(2.0)
        stopped=manager.stop(key)
        key2,note=manager.submit(AFTER_STOP,messages,memory,news_state,prior_key=key)
        if not key2 or key2==key:raise RuntimeError(f'New request not accepted: {note}')
        deadline=time.monotonic()+240
        while time.monotonic()<deadline:
            job=manager.get(key2)
            if job and job.phase in ('finished','cancelled','failed'):
                answer=job.snapshot()[0][-1]['content'];break
            time.sleep(.5)
        else:raise TimeoutError('No new answer within 240 s')
        rec=_record('10_stop',AFTER_STOP,answer,time.perf_counter()-t)
        rec['stop_accepted']=stopped
        rec['stop_phase']=manager.get(key).phase if manager.get(key) else 'missing'
        rec['new_answer_exact_437']=answer.strip()=='437'
    except Exception as exc:
        rec=_record('10_stop',AFTER_STOP,answer,time.perf_counter()-t,error=repr(exc))
    records.append(rec);_save(output_path,records,metadata)
    print('Saved:',output_path,'\nAll 10 cases attempted; any missing voice case is clearly marked.')
    print('No numerical 8/10 claimed. Human verification of the exact Qwen outputs is required.')
    return records


if __name__=='__main__':
    run_ten_case_acceptance()
