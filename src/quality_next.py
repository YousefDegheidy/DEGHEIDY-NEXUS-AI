"""Conservative structural QA and session-local repeat handling.
No semantic certification, cross-user cache or benchmark answer templates.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

DIGITS=str.maketrans('٠١٢٣٤٥٦٧٨٩','0123456789')
NUM_WORDS={'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'سبع':7,'ثمان':8,'اتنين':2,'ثلاث':3,'تلات':3,'أربع':4,'اربع':4,'خمس':5,'ست':6}

def normalized(s):return (s or '').translate(DIGITS)

def count(s, units):
    m=re.search(r'(?i)(\d+|one|two|three|four|five|six|اتنين|ثلاث(?:ة)?|تلات(?:ة)?|أربع(?:ة)?|اربع(?:ة)?|خمس(?:ة)?|ست(?:ة)?)\s*(?:'+units+r')',normalized(s))
    if not m:return None
    v=m[1].lower();return int(v) if v.isdigit() else NUM_WORDS.get(v) or NUM_WORDS.get(v.rstrip('ة'))

@dataclass(frozen=True)
class Spec:
    kind:str
    n:int
    tasks:int=0
    roles:int=0
    mitigations:int=0
    pages:bool=False
    memory:int=0
    measurable:bool=False
    numeric:bool=False
    deletion:bool=False
    incremental:bool=False
    access:bool=False
    per_user:bool=False


def spec_for(q):
    q=normalized(q)
    edit=re.search(r'(?i)(?:عدّل|عدل|غيّر|غير|modify|revise).{0,45}(?:الشهر|month)\s*(?:ال)?(أول|اول|ثاني|ثانى|ثالث|رابع|خامس|سادس|[1-6])',q)
    if edit and re.search(r'(?i)فقط|only|متعيدش|لا\s+تعيد',q):
        n=int(edit[1]) if edit[1].isdigit() else {'أول':1,'اول':1,'ثاني':2,'ثانى':2,'ثالث':3,'رابع':4,'خامس':5,'سادس':6}[edit[1]]
        t=count(q,r'مهام|مهمات|tasks?')
        if t and 2<=t<=8:return Spec('edit',n,t)
    if re.search(r'(?i)\bRAG\b',q) and re.search(r'(?i)architecture|صمم|صمّم|بنية\s*معمارية',q):
        n=count(q,r'خطوات|steps?')
        if n and 2<=n<=12 and re.search(r'(?i)hallucination|هلوس',q):
            m=count(q,r'طرق|وسائل|ways|methods') or 0
            mm=re.search(r'(?i)(?:آخر|اخر|last)\s*(\d+)\s*(?:رسائل|رسالة|messages?)',q)
            return Spec('rag',n,mitigations=m,pages=bool(re.search(r'رقم\s*(?:ال)?صفح|page\s*number',q,re.I)),memory=int(mm[1]) if mm else 0, deletion=bool(re.search(r'حذف|delete|remove',q,re.I)), incremental=bool(re.search(r'incremental|تحديث تدريجي',q,re.I)), access=bool(re.search(r'صلاحيات|access control|permissions',q,re.I)), per_user=bool(re.search(r'لكل مستخدم|per user',q,re.I)))
    if re.search(r'(?i)خطة|\bplan\b|roadmap',q) and re.search(r'(?i)وظيف|AI\s+Engineer|\bcareer\b',q):
        n=2 if re.search(r'شهرين|شهران',q) else count(q,r'شهور|أشهر|اشهر|months?')
        t=count(q,r'مهام|مهمات|tasks?')
        if n and t and 2<=n<=6 and 2<=t<=8:
            roles=2 if re.search(r'وظيفتين|وظيفتان|two\s+(?:roles|jobs)|2\s*(?:وظائف|roles|jobs)',q,re.I) else 0
            return Spec('plan',n,t,roles,measurable=bool(re.search(r'قابل[ة]?\s+للقياس|measurable',q,re.I)),numeric=bool(re.search(r'رقمي|رقمية|numeric|quantitative',q,re.I)))
    return None


def prompt_for(spec):
    if spec.kind=='edit':
        return (f'Write ONLY heading ## الشهر {spec.n}, followed by exactly {spec.tasks} separate numbered tasks. No other months, no job titles or extra sections. Preserve distinct projects and the latest user constraints. Do not repeat the old plan.')
    if spec.kind=='plan':
        return (f'Exact output format: {spec.n} headings, each on its own line: "## الشهر 1" and so on. '
                f'Under EACH heading put exactly {spec.tasks} tasks, each on a NEW line numbered 1. through {spec.tasks}. '
                'After each month put a separate line "النتيجة:" with a measurable target, not an achieved metric. '
                + (f'After all months write "## الوظائف" and exactly {spec.roles} numbered realistic paid junior titles. ' if spec.roles else '')+
                'No introductory text or other sections. Use existing projects where possible; no internship or invented experience. Specify a metric and unit, not undefined efficiency or guaranteed hiring. Targets are not measured results.')
    return (f'Use heading "## Architecture" and exactly {spec.n} numbered steps on separate lines. '
            'Explain ingestion, incremental updates and deletion, per-chunk metadata, indexing, retrieval, grounded generation; '
            + (f'explicitly store and use only the last {spec.memory} messages per session (separate from documents). ' if spec.memory else '')+
            ('Preserve document ID and page number during parsing and attach them to each retrieved chunk and citation. ' if spec.pages else '')+
            'Then heading "## الفرق" with separate explanations of RAG, Fine-tuning and Conversation Memory. '
            + (f'Then heading "## تقليل الهلوسة" with exactly {spec.mitigations} numbered techniques. ' if spec.mitigations else '')+
            ('Filter unauthorized chunks BEFORE retrieval or LLM context; isolate tenant/user IDs. ' if spec.access else '')+'Use the memory in query/context, invalidate obsolete document versions, never discard arbitrary last chunks. Abstain when evidence is absent; no invented confidence. No other sections; concise Egyptian Arabic. Do not claim implementation or guarantee correctness.')


def numbers(block):
    return [int(x) for x in re.findall(r'(?m)^\s*(?:[-*]\s*)?(?:\*\*)?(\d{1,2})[.)-]\s+',normalized(block))]


def violations(q,answer,spec=None):
    spec=spec or spec_for(q)
    if not spec:return []
    a=normalized(answer);errors=[]
    if not a.strip() or a.count('```')%2:errors.append('empty_or_unfinished')
    if spec.kind=='edit':
        heads=re.findall(r'(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:الشهر|شهر|month)\s*(\d+)\b',a)
        if heads!=[str(spec.n)]:errors.append('only_requested_month')
        if numbers(a)!=list(range(1,spec.tasks+1)):errors.append('edit_task_count')
        if re.search(r'(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:الوظائف|وظيفتان|وظيفتين|roles|jobs|خطة\s+العمل|جدول\s+زمني)',a):errors.append('unrequested_sections')
    elif spec.kind=='plan':
        heads=list(re.finditer(r'(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:الشهر|شهر|month)\s*(\d+)\b[^\n]*',a))
        if [int(x[1]) for x in heads]!=list(range(1,spec.n+1)):errors.append('monthly_headings')
        role=re.search(r'(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:الوظائف|وظيفتان|وظيفتين|roles|jobs)[^\n]*',a)
        if spec.roles and (not role or numbers(a[role.end():])!=list(range(1,spec.roles+1))):errors.append('role_count')
        for i,h in enumerate(heads):
            end=heads[i+1].start() if i+1<len(heads) else len(a)
            if role and h.end()<=role.start()<end:end=role.start()
            block=a[h.end():end]
            if numbers(block)!=list(range(1,spec.tasks+1)):errors.append(f'month_{i+1}_task_count')
            if spec.numeric and not re.search(r'(?im)^\s*(?:\*\*)?(?:النتيجة|المخرج|deliverable|result)\s*(?:\*\*)?\s*:[^\n]*\d',block):errors.append(f'month_{i+1}_numeric_target')
            if spec.measurable and not re.search(r'(?im)^\s*(?:\*\*)?(?:النتيجة|المخرج|deliverable|result)\s*(?:\*\*)?\s*:',block):errors.append(f'month_{i+1}_deliverable')
    else:
        diff=re.search(r'(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:الفرق|difference|comparison)',a)
        mitig=re.search(r'(?im)^\s*(?:#{1,6}\s*)?(?:\*\*)?(?:تقليل\s+(?:ال)?هلوس|طرق\s+تقليل|hallucination)',a)
        first=a[:diff.start()] if diff else a
        if numbers(first)!=list(range(1,spec.n+1)):errors.append('architecture_step_count')
        if not diff or not all(re.search(p,a[diff.start():mitig.start() if mitig else len(a)],re.I) for p in [r'\bRAG\b',r'fine[- ]?tuning',r'conversation\s+memory']):errors.append('concept_comparison')
        if spec.mitigations and (not mitig or numbers(a[mitig.end():])!=list(range(1,spec.mitigations+1))):errors.append('mitigation_count')
        if spec.memory and not re.search(rf'(?:آخر|اخر|last)\s*{spec.memory}\s*(?:رسائل|رسالة|messages?)',a,re.I):errors.append('explicit_memory_limit')
        if spec.pages and not re.search(r'رقم\s*(?:ال)?صفح|page\s*number',a,re.I):errors.append('page_citation')
        if spec.deletion and not re.search(r'(?i)(?:حذف|إزالة|احذف|delete|remove).{0,80}(?:ملف|مستند|قطع|فهرس|chunks?|documents?|files?|index)',first):errors.append('delete_obsolete_versions')
        if spec.incremental and not re.search(r'(?i)incremental|تحديث\s+تدريجي|القطع\s+المتغير',first):errors.append('incremental_indexing')
        if spec.access and not re.search(r'(?i)(?:صلاحيات|أذونات|ACL|permissions|access control).{0,90}(?:قبل|before)|(?:قبل|before).{0,90}(?:صلاحيات|أذونات|ACL|permissions|access control)',first):errors.append('access_before_retrieval')
        if spec.per_user and not re.search(r'(?i)لكل\s+(?:مستخدم|جلسة)|per[- ](?:user|session)',first):errors.append('per_user_isolation')
        if re.search(r'(?i)(?:استبعاد|حذف|تجاهل)\s+(?:آخر|اخر)\s*\d+\s*(?:قطع|chunks)',first):errors.append('arbitrary_chunk_exclusion')
    return list(dict.fromkeys(errors))


def repeat_answer(question, messages, memory):
    """Reuse only the immediately previous successful, identical static turn.
    No stale news, follow-ups about history, explicit retry requests or partial replies.
    """
    from src.realtime import needs_fresh_search
    from src.dialogue import is_followup, CORRECTION, GLOBAL_RECALL, evaluate_reply
    from src.coverage import missing_requirements
    q=question.strip()
    if needs_fresh_search(q) or is_followup(q) or CORRECTION.search(q) or GLOBAL_RECALL.search(q):return None
    if re.search(r'(?i)\b(?:latest|current|today|now|again|retry|random|another|different)\b|النهارده|اليوم|دلوقتي|حالي[اً]?|آخر\s+الأخبار|عيد\s+|مرة\s+تانية|مختلف|بديل|السعر|درجة\s+الحرارة',q):return None
    if len(messages)<2 or messages[-2].get('role')!='user' or messages[-2].get('content','').strip()!=q or messages[-1].get('role')!='assistant':return None
    a=messages[-1].get('content','')
    if not isinstance(a,str) or not a.strip() or a.startswith(('⚠️','⏳','✍️','⚙️','🌐','📰','🔎')) or 'تم إيقاف الرد' in a or 'لم يكتمل التحقق البنيوي' in a:return None
    # Require that the previous answer was actually committed by this session.
    if not any(t.get('role')=='assistant' and t.get('text')==a for topic in memory.get('topics',[]) for t in topic.get('turns',[])):return None
    if violations(q,a) or missing_requirements(q,a) or evaluate_reply(q,a,memory):return None
    return a


def structured_stream(q,memory,stream,build,budget,cancel=None, *, live_preview=False):
    """Preview static drafts in the UI; validate and attempt one repair.
    Only final text is committed. Non-UI callers retain final-only behavior.
    """
    spec=spec_for(q)
    assert spec is not None
    original=build(q,memory)
    messages=[dict(x) for x in original]
    messages[0]['content']+='\n\n'+prompt_for(spec)
    best='';issues=['empty_or_unfinished']
    limit=max(budget, min(1600, 300+spec.n*max(1,spec.tasks)*85))
    if spec.kind=='rag':limit=max(limit,1100)
    if spec.kind=='edit':limit=min(limit,650)
    preview_note='\n\n⏳ مسودة أولية — جاري التوليد وفحص اكتمال الشروط.'
    retry_note='\n\n⏳ المسودة تحتاج تصحيح بعض الشروط — جاري محاولة تصحيح واحدة.'
    partial_edit=None
    for attempt in range(2):
        if cancel is not None and cancel.is_set():return
        candidate=''
        print(f'[STRUCTURED] attempt={attempt+1}/2 | max_new_tokens={limit}',flush=True)
        if live_preview and attempt:
            yield (best+retry_note) if best else retry_note.strip()
        last_size=-1
        for chunk in stream(messages,min(limit,360) if partial_edit is not None else limit):
            if cancel is not None and cancel.is_set():return
            candidate=chunk
            if live_preview and candidate.strip() and len(candidate)!=last_size:
                last_size=len(candidate)
                if attempt==0:
                    yield candidate+preview_note
                else:
                    yield (best+retry_note if best else retry_note.strip())+f' (تم توليد {len(candidate)} حرف في التصحيح)'
        if cancel is not None and cancel.is_set():return
        if partial_edit is not None:
            old,expected=partial_edit
            if numbers(candidate)==expected and all(re.match(r'^\s*\d+[.)-]\s+\S',x) for x in candidate.strip().splitlines()):
                result=re.search(r'(?im)^\s*(?:\*\*)?(?:النتيجة|المخرج|deliverable|result)\s*(?:\*\*)?\s*:',old)
                candidate=(old[:result.start()].rstrip()+'\n'+candidate.strip()+'\n'+old[result.start():].lstrip()) if result else old.rstrip()+'\n'+candidate.strip()
            else:candidate=''
        found=violations(q,candidate,spec)
        print(f'[STRUCTURED] attempt={attempt+1} | violations={found}',flush=True)
        if candidate.strip() and (not best or len(found)<len(issues)):best,issues=candidate,found
        if not found:
            yield candidate;return
        messages=[dict(x) for x in original]
        nums=numbers(candidate)
        if attempt==0 and spec.kind=='edit' and found==['edit_task_count'] and nums==list(range(1,len(nums)+1)) and 0<len(nums)<spec.tasks:
            partial_edit=(candidate,list(range(len(nums)+1,spec.tasks+1)))
            messages[-1]['content']+='\n\nExisting valid tasks:\n'+candidate+'\nOutput ONLY missing numbered tasks '+str(partial_edit[1])+'. One line each; no heading, no repeats, no result. Keep projects separate.'
        else:
            messages[0]['content']+='\n\n'+prompt_for(spec)+'\nYour first draft failed these checks: '+', '.join(found)+'. Produce a fresh complete answer following the exact required format. Do not discuss these checks.'
            if attempt==0 and candidate.strip():messages[-1]['content']+='\n\nPrevious draft (revise, not evidence):\n'+candidate[:2400]
    if cancel is not None and cancel.is_set():return
    if not best:raise RuntimeError('Model produced no complete answer')
    yield best+'\n\n⚠️ لم يكتمل التحقق البنيوي من كل الشروط؛ راجع العدد والتفاصيل قبل الاعتماد.'
