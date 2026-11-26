"""Question-coverage planner and conservative, observable QA indicators.

Do not use these heuristic checks as factual verification or a guarantee that an LLM answer is correct.
"""
from __future__ import annotations
import re
from dataclasses import dataclass

@dataclass(frozen=True)
class Requirement:
    name: str
    instruction: str

_DURATION = re.compile(r'(?i)\bhow\s+(?:long|soon)\b|قد\s*(?:ايه|إيه)|المد[هة]|هاخد|هياخد|مدة|الوقت|وقت\s*قد')
_SENIOR = re.compile(r'(?i)\b(?:senior|sinior|seniour)\b|سنيور|سينيور|سينير')
_CAREER = re.compile(r'(?i)ai\s+eng(?:in|ne)|engineer|engneer|\bjob.ready\b|\bgpa\b|بقي?\s*سنيور|طالب\s+(?:سنة\s+رابعة\s+)?AI|وظيف|اتوظف|شغل\s+مدفوع')
_NEWS = re.compile(r'(?i)\b(?:news|headlines|politics)\b|أخبار|اخبار|خبرين|خبران|مستجدات|اخر\s+التطورات')
_EGYPT_IMPACT = re.compile(r'(?i)egypt|مصر|المصري')
_RUMOR = re.compile(r'(?i)\b(?:rumor|rumour|secret|trillion)\b|اشاعة|إشاعة|سري|تريليون')
_PYTHON = re.compile(r'(?i)\bpython\b|بايثون|\bfunction\b|دالة')
_TESTS = re.compile(r'(?i)\b(?:unit\s+tests?|pytest|unittest)\b|\b3\s+tests\b|اختبارات')
_OCCUPANCY = re.compile(r'(?i)occupancy|ركاب|passengers|أتوبيس|اتوبيس|seats|الإشغال|اشغال')
_MEMORY = re.compile(r'فاكر|قلتلك|قلت\s*لك|ذكرت(?:ها|ه)?|في\s*اول\s*كلامنا|remember|earlier\s+I\s+said',re.I)
_PLAN = re.compile(r'(?i)\b(?:plan|roadmap)\b|خط[هة]|خطة')
_SIX_MONTHS = re.compile(r'(?i)(?:6|٦|ست(?:ة)?|six)\s*(?:شهور|أشهر|اشهر|شهر|months?)')
_SINGLE_MONTH_EDIT = re.compile(r'(?i)(?:عدّل|عدل|غيّر|غير|modify|revise).{0,55}(?:الشهر|month).{0,15}(?:الأول|الاول|الثاني|الثالث|1|2|3|١|٢|٣)')
_THREE_MONTHS = re.compile(r'(?i)(?:3|٣|ثلاث(?:ة)?|تلات(?:ة)?)\s*(?:شهور|أشهر|اشهر|شهر|months?)|(?:three)\s+months?')
_THREE_TASKS_PER_MONTH = re.compile(r'(?i)(?:لكل\s+شهر|كل\s+شهر).{0,40}(?:3|٣|ثلاث|تلات)\s*(?:مهام|مهمات|tasks?)|(?:3|٣)\s*(?:مهام|tasks?).{0,40}(?:لكل\s+شهر|per\s+month)')
_TWO_ROLES = re.compile(r'(?i)(?:أول\s*)?وظيفتين|(?:2|٢|اتنين)\s*(?:وظائف|وظيف|فرص|roles?|jobs?)|two\s+(?:job\s+)?(?:roles?|jobs?)')


def extract_requirements(question: str) -> list[Requirement]:
    q=question or ''
    items=[]
    def add(n,s):
        if n not in {x.name for x in items}: items.append(Requirement(n,s))
    is_career=bool(_CAREER.search(q) and (_SENIOR.search(q) or 'GPA' in q.upper() or 'طالب' in q))
    is_news=bool(_NEWS.search(q))
    # The frequent three-month junior-AI roadmap was previously invisible to the
    # generic coverage planner (an English 'student' did not match Arabic طالب).
    # Detect the user's structural requirements independently from senior/GPA.
    if _PLAN.search(q) and _CAREER.search(q) and _THREE_MONTHS.search(q):
        add('career_three_month_plan',
            'FIRST complete month 1, month 2, and month 3, in that order. Each month has EXACTLY 3 short numbered actionable tasks and ONE measurable deliverable. Keep each task to one short sentence; no introductions, long examples or extra code.')
        if _THREE_TASKS_PER_MONTH.search(q):
            add('career_three_tasks', 'EXACTLY 3 tasks in EACH of the three months: 9 tasks total, plus 3 measurable monthly deliverables.')
        if _TWO_ROLES.search(q):
            add('career_two_roles', 'AFTER all three months, name EXACTLY TWO realistic PAID junior-level job titles for this user, not internships.')
        add('career_metrics_realistic', 'Use measurable targets/checkpoints, not made-up achieved accuracy, invented 40% speedups, or promises of employment. Do not confuse GitHub Pages (static site) with a deployed FastAPI backend.')
        add('career_format', 'FORMAT: شهر أول/ثان/ثالث: each has lines 1. 2. 3. and one **النتيجة:**. End with **وظيفتان مدفوعتان:** 1. and 2. No placeholders or unrequested months.')
    if _SINGLE_MONTH_EDIT.search(q):
        add('single_month_revision', 'Revise ONLY the specified month. Never include another month, additional future roadmap, or unsolicited integration between unrelated projects.')
        if re.search(r'(?i)(?:3|٣|ثلاث|تلات)\s*(?:مهام|tasks?)',q):
            add('single_month_three_tasks','Give exactly THREE numbered concrete tasks for the revised month, and nothing for other months.')
    if re.search(r'(?i)\bRAG\b|fine[- ]?tuning|فاين\s*تيون',q) and re.search(r'(?i)architecture|بنية\s*معمارية|هيكل|صمم|صمّم',q):
        add('rag_pipeline', 'RAG requires document ingestion/index update, query-time retrieval of RELEVANT chunks, generation with citations, and separately maintained recent-message memory. Do not treat last 5 chunks of a PDF as retrieval. Fine-tuning is not a memory store.')
        if re.search(r'(?i)(?:5|٥|خمس)\s*(?:خطوات|steps?)',q):
            add('rag_five_steps','Give FIVE distinct numbered architecture steps, including index update, relevant retrieval and source-cited answer.')
        if re.search(r'(?i)(?:3|٣|ثلاث|تلات)\s*(?:أخطاء|اخطاء|errors?|mistakes?)',q):
            add('rag_three_errors','After the pipeline include THREE distinct technical pitfalls, numbered 1 to 3.')
    if is_career:
        add('career_specific','Tailor advice to stated existing AI projects and limited company experience; do not present the user as a novice.')
        if _PLAN.search(q) and _SIX_MONTHS.search(q):
            add('career_roadmap','Provide months 1 through 6: ALL six individual month labels 1, 2, 3, 4, 5, 6 separately, one concise sentence each; do not group 1-2/3-4 or cut off month 6.')
        if re.search(r'(?i)job[- ]?ready|شغل\s+مدفوع|اشتغل|وظيف',q):
            add('career_job_ready','Separately estimate time to become competitive for PAID junior AI Engineer roles, as a conditional range, not a promise.')
        if _SENIOR.search(q) and _DURATION.search(q):
            add('career_senior_timeline','Separately explain Senior job title usually requires multi-year production responsibility, ownership and teamwork; six months of projects cannot guarantee Senior.')
        if re.search(r'(?i)\bgpa\b|المعدل|التقدير',q):
            add('career_gpa','Respect user request NOT to focus on improving GPA; do not recommend GPA improvement as the plan.')
        if re.search(r'اللي\s+ناقص|ناقصني|missing|gaps?',q,re.I):
            add('career_gaps','Describe possible professional gaps conditionally (test/verify), not pretend user disclosed lacking them.')
    if is_news:
        add('news_current','Present news with explicit source and publication/indexing times; do not claim headline index confirms event time.')
        if re.search(r'(?i)2\s*(?:أخبار|اخبار|news|headlines)|(?:أهم|اهم)\s*خبرين|خبرين|خبران',q):
            add('news_two','Give at most two relevant dated news items, or explicitly state fewer if only fewer verified sources exist.')
        if re.search(r'(?i)3\s*(?:أخبار|اخبار|news|headlines)|(?:أهم|top)\s*3',q):
            add('news_three','Summarize up to three DATED independent news items; if fewer, state how many are verified without inventing a third.')
        if re.search(r'(?i)trump|ترامب|تررامب',q):
            add('news_trump','Include a distinct section on the requested Trump/China developments using sources actually retrieved.')
        if _RUMOR.search(q):
            add('news_rumor','Explicitly examine the alleged secret 9-trillion agreement: confirmed, not supported by reviewed sources, or unverified; never call it false solely from no matches.')
        if _EGYPT_IMPACT.search(q):
            add('news_egypt','Separate CONDITIONAL Egypt impacts on Suez/shipping, imports/trade, energy and investment from proven meeting outcomes.')
        if re.search(r'(?i)24\s*(?:ساعة|hours?)|آخر\s*24',q):
            add('news_24h','Strict 24-hour lookback in Africa/Cairo (not the last three days).')
    if _MEMORY.search(q):
        add('memory','Recall only previously supplied USER facts; do not attribute unmentioned skills gaps as proven facts.')
    if _OCCUPANCY.search(q):
        add('occupancy','Calculate occupancy as passengers / capacity * 100 with explicit percentage and handle invalid capacity in code.')
    if _PYTHON.search(q) and re.search(r'(?i)\b(?:function|code|script|implement)\b|كود|دالة|اكتب|اكتبي|برنامج',q):
        add('python','Provide valid Python code that directly solves the requested task.')
    if _TESTS.search(q):
        add('tests','Include three distinct runnable unit tests for the supplied Python function.')
    if re.search(r'(?i)\b(?:operations?\s+research|\bOR\b)\b|بحوث\s+العمليات',q) and re.search(r'(?i)\bAI\b|الذكاء',q):
        add('ai_or','Distinguish demand prediction (AI) from assigning routes/vehicles with constraints (OR).')
    return items


def checklist_for_prompt(question: str) -> str:
    reqs=extract_requirements(question)
    if not reqs:return ''
    lines=['INTERNAL answer plan, not user-facing. Respond in ONE concise complete answer; no repeated introductions or follow-up drafts. Egyptian Arabic.']
    lines.extend(f'- {r.instruction}' for r in reqs)
    lines.append('Keep output compact: 1-2 sentences per requested section. If six months requested use 6 brief month bullets plus separate job-ready and senior timelines. Finish code fences and sentences. No repeated lists.')
    return '\n'.join(lines)


def missing_requirements(question: str, answer: str, *, prior_user_text: str='') -> list[str]:
    """Cheap under-answer signals; NOT a semantic answer grade."""
    if not answer.strip():return ['empty']
    a=answer.lower();missing=[]
    names={x.name for x in extract_requirements(question)}
    if re.search(r'(?i)(?:4|٤)\s*(?:خطوات|نقاط)|(?:أربع|اربعة|أربعة)\s*خطوات',question) and re.search(r'(?i)خطوة|الخطوة|steps?',answer):
        numbered=len(re.findall(r'(?m)^\s*(?:[1-9][.)-]|(?:الخطوة\s*[1-9٠-٩])\s*[:：])',answer))
        if numbered and numbered!=4:missing.append('exact_four_steps')
    if 'career_three_month_plan' in names:
        month_patterns=[
            r'(?i)(?:الشهر|شهر|month)\s*(?:ال?أول|ال?اول|1|١)',
            r'(?i)(?:الشهر|شهر|month)\s*(?:ال?ثاني|ال?ثانى|2|٢)',
            r'(?i)(?:الشهر|شهر|month)\s*(?:ال?ثالث|3|٣)',
        ]
        heads=[re.search(pattern,answer) for pattern in month_patterns]
        if not all(heads):
            missing.append('three_month_sections_incomplete')
        else:
            for idx,hit in enumerate(heads):
                block=answer[hit.end():(heads[idx+1].start() if idx<2 else len(answer))]
                # '3 tasks per month' requires THREE distinct numbered tasks,
                # not merely mention of the month or writing '3 tasks'.
                marker=re.search(r'(?i)وظيفت(?:ان|ين)|الوظائف\s+المناسبة|\bjobs?\b|\broles?\b',block)
                if marker and idx==2: block=block[:marker.start()]
                task_markers=re.findall(
                    r'(?m)^\s*(?:[-*]\s*)?(?:\*\*)?(?:(?:المهمة|مهمة|task)\s*[1-3١-٣]|[1-3١-٣][.)-])',
                    block,re.I)
                if 'career_three_tasks' in names and len(task_markers)<3:
                    missing.append(f'month_{idx+1}_tasks_incomplete')
                if not re.search(r'(?i)النتيجة|المخرج|\bdeliverable\b|\bmetric\b|قابل\s+للقياس',block):
                    missing.append(f'month_{idx+1}_deliverable_missing')
        if 'career_two_roles' in names:
            role_heading=re.search(r'(?i)وظيفتان|وظيفتين|الوظائف\s+المناسبة|وظائف\s+مدفوعة|two\s+(?:jobs?|roles?)',answer)
            roles_text=answer[role_heading.end():] if role_heading else ''
            # two numbered roles (or two distinct junior titles) after the plan
            num_roles=len(re.findall(r'(?m)^\s*(?:[-*]\s*)?(?:[12١٢][.)-])',roles_text))
            distinct_titles=len(set(re.findall(r'(?i)\bjunior\s+(?:AI|ML|machine learning|computer vision|LLM|NLP|MLOps)\s+(?:engineer|developer)\b',roles_text)))
            if num_roles<2 and distinct_titles<2: missing.append('two_roles_missing')
            if re.search(r'(?i)\binternships?\b|\btrainees?\b|تدريب\s+غير\s+مدفوع',roles_text):
                missing.append('prohibited_role_type')
    if 'single_month_revision' in names:
        requested=re.search(r'(?i)(?:الشهر|month)\s*(?:ال?أول|ال?اول|ال?ثاني|ال?ثالث|1|2|3|١|٢|٣)',question)
        if requested:
            def ord_index(s):
                if re.search(r'(?i)أول|اول|1|١',s): return 1
                if re.search(r'(?i)ثان|2|٢',s): return 2
                return 3
            keep=ord_index(requested.group(0))
            for n,pat in [(1,r'(?i)(?:الشهر|شهر|month)\s*(?:ال?أول|ال?اول|1|١)'),
                          (2,r'(?i)(?:الشهر|شهر|month)\s*(?:ال?ثاني|ال?ثانى|2|٢)'),
                          (3,r'(?i)(?:الشهر|شهر|month)\s*(?:ال?ثالث|3|٣)')]:
                if n!=keep and re.search(pat,answer):missing.append('unrequested_months')
    if 'rag_five_steps' in names:
        rag_block=re.split(r'(?i)أخطاء|اخطاء|errors?|mistakes?',answer,maxsplit=1)[0]
        steps=re.findall(r'(?m)^\s*(?:[-*]\s*)?(?:[1-5١-٥][.)-])',rag_block)
        if len(steps)<5: missing.append('rag_steps_incomplete')
    if 'rag_three_errors' in names:
        m=re.search(r'(?i)أخطاء|اخطاء|errors?|mistakes?',answer)
        error_block=answer[m.end():] if m else ''
        if len(re.findall(r'(?m)^\s*(?:[-*]\s*)?(?:[1-3١-٣][.)-])',error_block))<3:
            missing.append('rag_errors_incomplete')
    if 'career_roadmap' in names:
        # Allow e.g. شهر 6 / الشهر السادس / Month 6.
        if not re.search(r'(?i)(?:الشهر|شهر|month)\s*(?:ال?سادس|6|٦)|(?:6|٦)\s*(?:-|–|:)\s*(?:شهر|month)',a): missing.append('six_month_roadmap')
    if 'career_job_ready' in names and not re.search(r'(?i)job[- ]?ready|junior|جاهز|للتوظيف|تقديم|وظيف|شغل',a): missing.append('job_ready')
    if 'career_senior_timeline' in names:
        if not (re.search(r'(?i)senior|سنيور|سينير|سينيور',a) and re.search(r'(?i)سن(?:ة|ين|وات)|عام|years?',a)):
            missing.append('senior_timeline')
    if 'news_rumor' in names and not re.search(r'(?i)9\s*(?:trillion|تريليون)|تسع(?:ة)?\s+تريليون|الاتفاق\s+السري|اتفاق\s+سري',a): missing.append('rumor')
    if 'news_egypt' in names and not re.search(r'قناة\s+السويس|suez',a,re.I):missing.append('egypt_impact')
    if 'memory' in names and not re.search(r'(?i)tensorflow|pytorch|computer vision|مشاريع|مشروعات|gpa',a): missing.append('memory')
    if 'occupancy' in names and not re.search(r'(?:\d+(?:\.\d+)?\s*(?:%|٪|بالمئة|بالمائة))',a):missing.append('occupancy_percent')
    if 'python' in names and not re.search(r'```python|\bdef\s+\w+\s*\(',a,re.I): missing.append('python_code')
    if 'tests' in names and not (len(re.findall(r'(?m)^\s*def\s+test_',answer))>=3 or len(re.findall(r'(?m)^\s*assert\b',answer))>=3):missing.append('three_tests')
    if 'ai_or' in names:
        has_ai=bool(re.search(r'(?i)\bAI\b|تنبؤ|الذكاء|machine learning',answer))
        has_or=bool(re.search(r'(?i)\bOR\b|optimization|تحسين|برمجة خطية|تخصيص',answer))
        if not (has_ai and has_or):missing.append('ai_or')
    from src.dialogue import is_cutoff
    if is_cutoff(answer) or (answer.count('```') % 2):missing.append('unfinished_text')
    return missing
