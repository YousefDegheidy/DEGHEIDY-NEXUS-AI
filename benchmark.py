"""Real-model Colab regression run; NEVER silently invent a quality score.

Eight text questions are sent to the real Qwen3.5 app.turn path on GPU. Audio
quality, Stop UI and browser connectivity still require hands-on verification.
The JSON includes complete text, measured times, and non-semantic automated flags.
"""
from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json,re,time
from src.coverage import missing_requirements,extract_requirements
from src.dialogue import fresh_memory
from src.realtime import should_search

CASE_PROMPTS=[
 ('buses','عندي 3 أتوبيسات: الأول 42 كرسي وفيه 33 راكب، الثاني 30 كرسي وفيه 24 راكب، والثالث 18 كرسي وفيه 9 ركاب. احسب نسبة إشغال كل أتوبيس والإجمالي للأسطول كله، ورتّبهم من الأقل للأعلى. وضح الحسابات.'),
 ('python','عندي الكود ده:\ndef average(numbers):\n    return sum(numbers) / len(numbers)\nاشرح المشكلة لما القائمة تكون فاضية، وصلح الدالة، واكتب 3 اختبارات pytest للحالة الطبيعية والفارغة والسالبة.'),
 ('career','أنا طالب AI وعندي مشاريع في الرؤية الحاسوبية وLLMs، ومش عايز أبدأ من الصفر. اقترح خطة عملية لأول 30 يوم لبرنامج توظيف مدفوع، في 4 خطوات مرقمة فقط. متقترحش Internships ولا كورسات مبتدئين ولا تفترض إني اشتغلت في Production.'),
 ('architecture','عندي تطبيق CNN لصور النباتات. صمّم Architecture من 5 مكونات للنشر باستخدام FastAPI وDocker، ووضح Latency وError Rate وModel Drift وإيه أعمل لو السيرفر وقع. مختصر.'),
 ('memory','فاكر أنا قلتلك في بداية كلامنا عندي خبرة ومشاريع في إيه؟ وإيه القيود اللي حددتها لما طلبت خطة شغل؟ افصل المعلومات اللي قلتها أنا عن اللي إنت استنتجتها.'),
 ('recent_news','اختار أهم خبرين دوليين نُشرا خلال آخر 12 ساعة بتوقيت القاهرة. لكل خبر ماذا حدث وتاريخ الحدث وتاريخ النشر والمصدر الأصلي ورابطه، واستبعد الرأي.'),
 ('rumor','قرأت منشورًا بيقول إن Google قررت إيقاف Gemini نهائيًا الأسبوع الجاي. هل فيه إعلان رسمي يؤكد ده؟ تحقق من مصادر موثوقة؛ متخترعش بيان رسمي.'),
 ('mixed','I built an AI chatbot using Python and Hugging Face، بس الـresponses ساعات slow وبيحصل context loss. Explain بالمصري إزاي أشخص latency والذاكرة واديني 3 خطوات لتحسين الأداء من غير ما أضحي بجودة الإجابات.'),
]


def objective_flags(name, answer):
    """Mechanical indicators, not independent factual accuracy or human quality scores."""
    a=answer.lower();flags={}
    if name=='buses':
        flags={k:k in a for k in ('78.57%','80%','50%','73.33%','66 ÷ 90','الثالث (50%)')}
    elif name=='python':
        flags={'empty_case':'ValueError' in answer or 'None' in answer,
               'three_tests':len(re.findall(r'(?m)^\s*def test_',answer))>=3,
               'code_fences_closed':answer.count('```')%2==0}
    elif name=='career':
        flags={'references_four_steps':bool(re.search(r'4\s*خطوات|أربع\s*خطوات|(?:^|\n)\s*4\.',a)),
               'no_internship_recommendation':not bool(re.search(r'(?i)(?:قدّم\s+على\s+internship|أنصحك\s+بتدريب|اعمل\s+internship)',a))}
    elif name=='architecture':
        flags={'latency':bool(re.search(r'(?i)latency|زمن\s+الاستجابة',a)),
               'error_rate':bool(re.search(r'(?i)error\s+rate|معدل\s+الأخطاء',a)),
               'drift':bool(re.search(r'(?i)model\s+drift|انحراف\s+النموذج',a))}
    elif name=='memory':
        flags={'llms':'llms' in a,'vision':'الرؤية\u0020الحاسوبية' in a or 'computer vision' in a,
               'no_unfounded_denial':not bool(re.search(r'لم\s+تذكر\s+أي\s+(?:مشروع|خبرة)|لم\s+تقل\s+ل[يي]\s+عن\s+مشاريع',a))}
    elif name=='recent_news':
        flags={'has_links':'http' in a,'not_editorial':not bool(re.search(r'(?i)^\s*\d+\.\s*.*(?:رأي:|opinion:)',a,re.M)),
               'no_wrong_date_on_empty':'2026-11-19' not in a}
    elif name=='rumor':
        flags={'acknowledges_uncertainty':bool(re.search(r'(?i)غير\s+مثبت|غير\s+مؤكد|لم\s+أجد|لا\s+أستطيع\s+تأكيد|لا\s+يوجد\s+دليل',a)),
               'no_claim_of_fake_investigations':'أثبتت التحقيقات' not in a}
    elif name=='mixed':
        flags={'latency':bool(re.search(r'(?i)latency|البطء|سرعة',a)),
               'memory':bool(re.search(r'(?i)memory|context|الذاكرة|السياق',a))}
    return flags


def run_live_benchmark(output_path='/content/DEGHEIDY_REAL_QWEN_8_TEXT_TESTS.json'):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Real Qwen test was NOT run: CUDA GPU unavailable. Enable T4 in Google Colab. No 8/10 score may be asserted.')
    import app
    memory=fresh_memory();messages=[];news_state=None;records=[]
    path=Path(output_path)
    for i,(label,question) in enumerate(CASE_PROMPTS,1):
        begin=time.perf_counter();first=None;last=None;frames=0
        print(f'[{i}/8] {label}: running real Qwen/app path...',flush=True)
        try:
            routed=should_search(question,memory)
            for snap in app.turn(question,messages,memory,news_state):
                last=snap;frames+=1
                text=snap[0][-1]['content'] if snap[0] else ''
                if first is None and text.strip() and not text.startswith(('⏳','🌐','📰','⚙','✍','🔎')):
                    first=time.perf_counter()-begin
            if not last:raise RuntimeError('No frames returned')
            messages,_,memory,news_state=last
            answer=messages[-1]['content']
            bad=answer.startswith(('⚠️','الشات المحلي محتاج GPU'))
            rec={'case':label,'question':question,'real_gpu_model_requested':True,
                 'route':'NEWS' if routed else 'CHAT','duration_sec':round(time.perf_counter()-begin,2),
                 'first_content_sec':round(first,2) if first is not None else None,
                 'frames':frames,'answer':answer,'runtime_error':bad,
                 'flags_for_review':objective_flags(label,answer),
                 'missing_heuristics':missing_requirements(question,answer)}
        except Exception as err:
            rec={'case':label,'question':question,'runtime_error':True,'error':repr(err)}
        records.append(rec)
        path.write_text(json.dumps({'created':datetime.now().isoformat(),
            'scope':'8 real-model text tests if GPU is available; not a calibrated quality score',
            'manual_still_needed':['speech WER/CER','Stop after cancel','reload recovery','source content verification'],
            'result':'NOT CERTIFIED 8/10: human evaluation required',
            'records':records},ensure_ascii=False,indent=2),encoding='utf8')
        print(f'[{i}/8] saved {rec.get("duration_sec","ERROR")}s, {len(rec.get("answer",""))} chars, flags={rec.get("flags_for_review")}',flush=True)
    print('Saved real-model results:',path,'— no numeric 8/10 claim without independent answer review')
    return records
