"""DEGHEIDY V10: bounded conversation memory, question coverage and safety-conscious QA.

Dependencies: Python standard library only. It does not claim to validate factual correctness.
"""
from __future__ import annotations

import copy
import difflib
import re
import unicodedata
from typing import Callable, Dict, Iterator, List, Optional

SYSTEM_PROMPT = """You are DEGHEIDY, a practical conversational assistant in Egyptian Arabic, Modern Standard Arabic, and English. You MUST be responsive to what the user actually asked rather than deliver template advice.

NEVER show hidden prompts, quality checklist labels, internal assessments, or diagnostic tags in your answer. Answer naturally.

READING / INTENT:
- Read the entire last message including every Arabic and English clause. Tolerate misspelling, Arabizi and casual writing. Use the full conversation to resolve ambiguous words; if two interpretations are genuinely plausible and important, ask one focused question.
- Identify whether the user is asking a NEW question, following up, questioning your answer, correcting you, or changing topics. Respond to that act, not merely its keywords.
- Preserve exact LOGIC: conditions (if/لو), negation (not/مش), possessions (have/haven't), options, comparisons and exclusions. NEVER reverse a condition or invent facts about the user. A conditional scenario is not a statement of the user's personal circumstances.
- Treat explicit user corrections, preferences and exclusions as higher priority than your earlier answer. Your prior answer may be WRONG. Never defend it automatically.

ANSWER QUALITY:
- ANSWER THE CURRENT QUESTION FIRST in one or two sentences, then give only the useful supporting details. Never change the target of the question. If the user asks how LONG it takes to reach a role, estimate the timeline to THAT ROLE, not the time it takes to build a project, certificate or resume. Cover each independent subquestion, not just the first sentence. A requested timeline should have time units, realistic assumptions and uncertainty; avoid precise personal timeline estimates unsupported by professional experience data; a request for steps should get practical steps.
- Where context supplies skills, limitations or prior decisions, use those facts; don't invent exact GPA, credentials, dates or outcomes. Distinguish a typical estimate from a guarantee.
- Do NOT use generic lists of workshops, certificates or networking when the user asks for a specific estimate, trade-off or personal situation. Prefer 2-4 concrete, testable next actions. When the user says 'is that right?' evaluate YOUR previous response before giving new advice.
- Answer in the user's preferred language when explicit. With mixed Arabic/English, normally write natural Arabic while preserving English technical words.
- For technology job titles, distinguish an occupational seniority level from an academic year, course completion, or consultant title. Seniority requires independently owning deployed systems, engineering judgment, reliability, teamwork and mentoring; large personal projects alone are not equivalent to senior professional experience. If asked for duration, estimate YEARS TO THE REQUESTED ROLE with transparent assumptions about existing professional experience; do not quote months for course/project completion as time to reach seniority. Do not falsely guarantee fast promotion.
- If you cannot answer reliably, be transparent about uncertainty; do not make up facts. Follow safety constraints and do not refuse ordinary benign requests without a relevant reason.
- When asked for the fastest path, provide a realistic range, an explicit distinction between skill level and job title, and the highest-leverage actions. If past user context indicates completed projects/courses, do not instruct them to restart as a beginner. Prioritize real production deployment, testing, monitoring, failure recovery, system design, and collaboration, when relevant.
- Prefer short, useful answers and finish all sentences. Do not dump 10 generic tips or start a numbered item you cannot finish. Never use Chinese characters or unrelated languages in Arabic/English replies.
"""

ARABIC = re.compile(r'[\u0600-\u06ff]')
ENGLISH = re.compile(r'[A-Za-z]')
# Match common spelling corrections conservatively. Never autocorrect names or arbitrary unknown words.
KNOWN_TYPOS = {
    'sinior': 'senior', 'seniour': 'senior', 'enffnerr': 'engineer',
    'engneer': 'engineer', 'enginner': 'engineer', 'aai': 'AI',
    'pyhton': 'Python', 'hhow': 'how', 'hwo': 'how', 'responce': 'response',
}
SYNONYMS = {
    'سينيور':'senior', 'سنيور':'senior', 'سينير':'senior', 'سينيير':'senior', 'ذكاء':'ai', 'اصطناعي':'ai', 'الاصطناعي':'ai',
    'صناعي':'ai','الذكاء':'ai', 'مشروع':'project', 'مشاريع':'project',
    'بروجيكت':'project','بروجيكتات':'project', 'projects':'project',
    'مهندس':'engineer','هندسة':'engineer','skills':'skill', 'مهارات':'skill',
    'خبرة':'experience','experience':'experience','بايثون':'python',
    'سنين':'years', 'سنة':'years','سنوات':'years',
}
STOP_WORDS = set('''the a an and or i me my he she it its is are to from about your you
in on of by at for with as that this what when how why but can does do
انا انت انتو نحن هو هي ده دي دول اللي اللى على في من عن و او بس طب طيب
ما مش مع لو معاك عندي عندك عنده ممكن عايز لو عاوز هيبقى هيبقي
'''.split())

CORRECTION = re.compile(
    r"(?i)^\s*(?:no\b|nope\b|not\b|actually\b|i\s+(?:don't|dont|didn't|didnt|meant|said|ont)\b|"
    r"you\s+(?:missed|misunderstood|forgot)\b|but\b|"
    r"لأ\b|لا\b|مش\s+قصدي|قصدي\b|أنا\s+قلت|انا\s+قلت|مش\s+عايز|لكن\b|بل\b)"
)
FOLLOWUP = re.compile(
    r"(?i)^\s*(?:so\b|then\b|what\s+about\b|and\b|but\b|"
    r"is\s+(?:that|this)\b|does\s+(?:this|that)\b|"
    r"طب\b|طيب\b|يعني\b|و\s*لو\b|بخصوص\b|بالنسبة\b|كده\b|بس\b)"
)
TOPIC_RESET = re.compile(r'(?i)\b(?:new topic|change (?:the )?topic|different topic|anyway)\b|موضوع\s+تاني|نغير\s+الموضوع|بالمناسبة|بعيد[اً] عن')
REFERENTIAL = re.compile(r'(?i)\b(?:it|this|that|these|them|those|same|previous|earlier)\b|ده|دا|دي|دول|اللي\s+فات|نفس|قبل\s+كده')
TIMELINE_REQUEST = re.compile(
    r'(?i)\bhow\s+(?:long|soon|much\s+time)\b|\btime(?:frame|line)?\b|'
    r'قد\s*(?:إيه|ايه|اه)|كام\s*(?:سنة|سنين|شهر|وقت)|\bوقت\b|\bمدة\b|'
    r'المدة|المده|أقل\s*وقت|اقل\s*وقت|هي(?:ا|ي)خد\s*قد|ياخد\s*قد|'
    r'يستغرق|كم\s*(?:يستغرق|سنة|شهر|وقت)'
)
HOW_REQUEST = re.compile(r'(?i)\bhow\s+(?:to|can|should|do|could)\b|إزاي|ازاي|ازاى|كيف|الخطوات')
WHY_REQUEST = re.compile(r'(?i)\bwhy\b|ليه|لماذا|السبب')
COMPARE_REQUEST = re.compile(r'(?i)\b(?:compare|comparison|versus|vs\.?|better\s+than|difference)\b|قارن|مقارنة|الفرق\s+بين|أنهي\s+أحسن')
EVAL_REQUEST = re.compile(r'(?i)^\s*(?:so\s+is\s+(?:that|this)|so\s+(?:this|that)\s+is|is\s+(?:that|this)|does\s+(?:that|this)|يعني\s+(?:ده|كده)|ده\s+كده|يعني\s+دي)')
YESNO_REQUEST = re.compile(r'(?i)^\s*(?:do|does|can|could|is|are|will|would|should)\b|(?:^|[\s،])هل\s')
EXCLUSION = re.compile(r"(?i)(?:i\s+(?:don't|dont|do\s+not)\s+want\s+to|مش\s+عايز|لا\s+أريد|لا\s+اريد)\s+([^,.!?\n]{4,70})")
ADVICE = re.compile(r'(?i)(?:you\s+(?:should|need\s+to)|i\s+recommend|try\s+to|لازم|أنصحك|ينبغي\s+أن)\s+([^.!?\n]{4,100})')
# Numbers in numbered lists do NOT count as timeframe answers.
DURATION = re.compile(
    r'(?i)(?<![\d٠-٩])[\d٠-٩]+(?:[.,][\d٠-٩]+)?'
    r'(?:\s*(?:-|–|—|to|إلى|الى)\s*[\d٠-٩]+(?:[.,][\d٠-٩]+)?)?'
    r'\s*(?:سنوات?|سنة|سنه|سنين|شهور?|شهر|أشهر|عام|أعوام|أسبوع|أسابيع|يوم|أيام|'
    r'years?|months?|weeks?|days?|hours?|ساعات?|ساعة)\b'
)
SLASH_DURATION = re.compile(
    r'(?i)\d+\s*(?:سنوات?|سنة|سنه|سنين|أشهر|شهر|years?|months?)\s*/\s*'
    r'\d+\s*(?:سنوات?|سنة|سنه|سنين|أشهر|شهر|years?|months?)'
)
REFUSAL = re.compile(
    r"(?i)(?:i(?:'m| am)\s+sorry.{0,60}(?:can't|cannot|unable)|"
    r"i\s+(?:can't|cannot|won't)\s+(?:assist|help)\s+with\s+that|"
    r'لا\s+(?:أستطيع|استطيع|يمكنني)\s+(?:مساعدتك|المساعدة))'
)
BENIGN = re.compile(r'(?i)\b(?:learn|study|career|engineer|senior|sinior|ai|python|project|work|skills|gpa|college)\b|تعلم|مهندس|مشروع|مهارة|جامعة|وظيفة|ذكاء')
RISKY = re.compile(r'(?i)\b(?:weapon|bomb|explosive|poison|suicide|steal|fraud|hacking|malware|bypass)\b|قنبلة|سلاح|انتحار|سرقة|احتيال|اختراق')

# Work with both Arabic and English syntax while avoiding specific canned answers.
POSSESSION = re.compile(r'(?i)(?:\b(?:if\s+you\s+have|you\s+have|i\s+have|having)\s+|(?:لو\s+)?(?:معاك|معايا|عندك|عندي|لديك|معي)\s+)([^،,.?!\n]{2,65})')
NEGATIVE_POSSESSION = re.compile(r"(?i)\b(?:don't\s+have|do\s+not\s+have|without|lack|lacking|haven't\s+got)\b|\b(?:معندكش|معنديش|مش\s+عندك|لا\s+تملك|ليس\s+لديك|ليس\s+عندك|بدون)\b")
NEGATED_PREMISE = re.compile(r'(?i)(?:اذا|إذا|لو|if|since|because|بما\s+انك|بما\s+أنك).{0,28}(?:معندكش|لا\s+تملك|ليس\s+عندك|ليس\s+لديك|don.t\s+have|do\s+not\s+have|without|lack)',re.S)
SENIOR = re.compile(r'(?i)\b(?:senior|seniour|sinior|principal|staff\s+engineer)\b|سينيور|سنيور|سينير|سينيير')
ROLE_DRIFT = re.compile(r'(?i)(?:become\s+(?:an?\s+)?(?:ai\s+)?(?:consultant|advisor)|تصبح\s+مستشار|التحول\s+إلى\s+مستشار|تتحول\s+إلى\s+مستشار)')
GENERIC_TERMS = re.compile(r'(?i)\b(?:workshops|networking|conferences|certifications|webinars|courses)\b|ورش\s+عمل|مؤتمرات|شهادات|كورسات')


def lookup_normalize(message: str) -> str:
    """Loose matching for noisy Arabic text; never sent as a replacement for user text."""
    t = unicodedata.normalize('NFKC', str(message or '')).lower()
    t = re.sub(r'[\u064b-\u065f\u0670\u0640]', '', t)
    t = t.translate(str.maketrans('أإآى', 'اااي'))
    # Avoid aggressively changing doubles in normal Arabic; collapse only 3+ repeats.
    t = re.sub(r'([\u0621-\u064a])\1{2,}', r'\1', t)
    t = re.sub(r'ا+ي+ه', 'ايه', t)
    return t


def spelling_hints(message: str) -> str:
    """Conservative hints passed alongside original text, never changing its meaning."""
    replacements=[]
    for wrong,right in KNOWN_TYPOS.items():
        if re.search(r'\b'+re.escape(wrong)+r'\b', message, re.I):
            replacements.append(f'{wrong} → {right}')
    return ', '.join(replacements)


def normalized_tokens(message: str) -> set[str]:
    t=unicodedata.normalize('NFKC',str(message or '')).lower()
    t=re.sub(r'[\u064b-\u065f\u0670\u0640]','',t)
    t=t.translate(str.maketrans('أإآى','اااي'))
    words=re.findall(r'[a-z0-9]+|[\u0621-\u064a]+',t)
    return {SYNONYMS.get(KNOWN_TYPOS.get(w,w).lower(),KNOWN_TYPOS.get(w,w).lower()) for w in words
            if w not in STOP_WORDS and len(w)>1}


def relatedness(message: str, topic: dict) -> float:
    q=normalized_tokens(message)
    user_turns=[t['text'] for t in topic.get('turns',[]) if t['role']=='user'][-6:]
    prev=set().union(*(normalized_tokens(x) for x in user_turns)) if user_turns else set()
    common=len(q & prev)
    fuzzy=sum(.45 for w in q-prev if len(w)>5 and any(len(x)>5 and difflib.SequenceMatcher(None,w,x).ratio()>.86 for x in prev))
    return common+fuzzy


def fresh_memory() -> dict:
    return {'topics': [], 'active': None, 'clock': 0}


def is_followup(message: str) -> bool:
    return bool(FOLLOWUP.search(message) or CORRECTION.search(message) or EVAL_REQUEST.search(message) or
                (REFERENTIAL.search(message) and len(normalized_tokens(message)) < 7) or
                (re.match(r'(?i)^\s*(?:لو|if)\b', message) and wants_time(message)))


def pick_topic(message: str, memory: dict) -> Optional[int]:
    topics=memory.get('topics',[])
    if not topics or TOPIC_RESET.search(message):return None
    active=memory.get('active')
    scores=[relatedness(message,t) for t in topics]
    candidate=max(range(len(scores)), key=lambda i:scores[i])
    score=scores[candidate]
    # An old topic may be revived when there is strong lexical evidence.
    if active is not None and candidate != active and score>=2:
        return candidate
    if active is not None:
        # An explicit correction or a question about a previous answer always refers back.
        if CORRECTION.search(message) or EVAL_REQUEST.search(message):return active
        active_score=scores[active]
        if active_score>=1:return active
        # Short elliptical followups inherit the active topic. A substantial new subject
        # with no overlap should not inherit it just because it begins with 'طب'.
        if is_followup(message):
            # "طب لو معايا مشاريع... هاخد قد ايه؟" is a continuation even when
            # its terms don't overlap with the original "senior AI" goal.
            # Don't inherit a completely new how-to just because it begins "طب".
            if (REFERENTIAL.search(message) or wants_time(message) or
                    (is_conditional(message) and len(normalized_tokens(message))<=10)):
                return active
            if len(normalized_tokens(message))<=3 and not HOW_REQUEST.search(message):
                return active
    if score>=2:return candidate
    return None


def is_mixed(message:str)->bool:
    return bool(ARABIC.search(message) and ENGLISH.search(message))


def is_conditional(message:str)->bool:
    return bool(re.search(r'(?i)\bif\b|\bunless\b|(?:^|[\s،])لو(?:[\s،]|$)|\bإذا\b|\bاذا\b',message))


def wants_time(message:str)->bool:
    raw = str(message or '')
    lookup = lookup_normalize(raw)
    return bool(TIMELINE_REQUEST.search(raw) or
                re.search(r'(?:قد\s*ايه|كام\s*(?:سنه|شهر|وقت)|\bhow\s+long\b|\bمدة\b|\bالمدة\b|هاخد|ياخد|هتاخد|هياخد|ف\s*اقل\s*وقت)', lookup, re.I))


def user_profile_notes(memory: dict, idx: Optional[int])->List[str]:
    if idx is None:return []
    turns=memory['topics'][idx]['turns']
    user=[t['text'] for t in turns if t['role']=='user']
    # Literal, attributable excerpts instead of a generated summary that can invent facts.
    return user[-6:]


def requested_aspects(question:str)->List[str]:
    checks=[]
    if HOW_REQUEST.search(question):checks.append('ACTION: give applicable practical steps, not generic slogans.')
    if wants_time(question):checks.append('TIME: answer how long with a plausible time range, units and conditions (not just a numbered list).')
    if WHY_REQUEST.search(question):checks.append('REASON: give a causal explanation.')
    if COMPARE_REQUEST.search(question):checks.append('COMPARISON: explain the relative differences.')
    if is_conditional(question):checks.append('CONDITION: preserve the exact stated IF / لو premise and its positive/negative polarity. Never reverse it.')
    if CORRECTION.search(question):checks.append('CORRECTION: update your response according to the user, not your prior answer.')
    if EVAL_REQUEST.search(question):checks.append('EVALUATION: explicitly say whether your prior answer was right, partly right or wrong, and what it missed.')
    if YESNO_REQUEST.search(question):checks.append('DIRECT DECISION: start with an explicit yes/no/it depends answer where appropriate, then explain the actual condition.')
    if EXCLUSION.search(question):checks.append('EXCLUSION: do NOT recommend the action that the user explicitly said they do not want to do.')
    if SENIOR.search(question):checks.append('JOB LEVEL: do not confuse a job seniority level with a consultant title or a university year.')
    if is_mixed(question):checks.append('MULTILINGUAL: cover BOTH English and Arabic clauses in the same answer.')
    return checks


def build_messages(question:str,memory:dict)->list[dict]:
    question=question.strip()
    if not question:raise ValueError('Empty question')
    memory=memory or fresh_memory()
    idx=pick_topic(question,memory)
    turns=(memory['topics'][idx]['turns'] if idx is not None else [])
    prompts=[SYSTEM_PROMPT]
    hints=spelling_hints(question)
    if hints:prompts.append('Spelling interpretation hints, NOT a replacement for the user message: '+hints)
    aspects=requested_aspects(question)
    if aspects:prompts.append('MANDATORY parts of the LAST question (answer ALL of them):\n- '+'\n- '.join(aspects))
    if is_conditional(question):prompts.append('USER PREMISE quoted EXACTLY: '+question[:500]+
        '\nPreserve the polarity of all conditions. In a conditional scenario, do not rewrite "has" as "does not have".')
    if turns:
        first_user=next((t['text'] for t in turns if t['role']=='user'), '')
        last_user=[t['text'] for t in turns if t['role']=='user'][-5:]
        # Goal anchor is sourced ONLY from prior user words, not model-generated guesses.
        prompts.append('ACTIVE TOPIC GOAL — earliest USER question, quoted exactly:\n'+first_user[:500]+
            '\nThe LAST question is a follow-up to this goal unless the user explicitly changes targets. '
            'Keep the object of a duration/decision question anchored to the original goal. '
            'For example, a project can be a prerequisite/condition, not a new career target.')
        prompts.append('Other recent USER statements / corrections; later statements override older ones:\n'+
                       '\n'.join('- '+x[:220] for x in last_user))
        if wants_time(question):
            prompts.append('DURATION TARGET: derive the thing the user wants to achieve from the active-topic goal, '
                           'NOT from a mentioned supporting activity. State what your duration estimate refers to.')
    if CORRECTION.search(question):
        prompts.append('User is CORRECTING your previous answer. Explicitly acknowledge what was wrong, then answer the revised request. Do not repeat rejected advice.')
    recent=turns[-8:]
    if CORRECTION.search(question):
        # This remains a copy: the stored memory retains history for evaluation.
        last_ai=next((i for i in range(len(recent)-1,-1,-1) if recent[i]['role']=='assistant'),None)
        if last_ai is not None:
            prompts.append('Your previous assistant response was rejected and is not evidence. Your next reply must change accordingly.')
            recent=recent[:last_ai]+recent[last_ai+1:]
    messages=[{'role':'system','content':'\n\n'.join(prompts)}]
    messages.extend({'role':x['role'],'content':x['text'][:750]} for x in recent)
    messages.append({'role':'user','content':question})
    return messages


def commit_answer(memory:dict,question:str,answer:str)->dict:
    m=copy.deepcopy(memory or fresh_memory())
    idx=pick_topic(question,m)
    if idx is None:
        if len(m['topics'])>=8:
            m['topics'].pop(0)
            if m['active'] is not None:m['active']=max(m['active']-1,0)
        m['topics'].append({'turns':[],'last':0})
        idx=len(m['topics'])-1
    m['clock']+=1
    m['topics'][idx]['turns'] += [{'role':'user','text':question},{'role':'assistant','text':answer}]
    m['topics'][idx]['turns']=m['topics'][idx]['turns'][-24:]
    m['topics'][idx]['last']=m['clock']
    m['active']=idx
    return m


def previous_answer(question,memory):
    idx=pick_topic(question,memory or fresh_memory())
    if idx is None:return ''
    return next((x['text'] for x in reversed(memory['topics'][idx]['turns']) if x['role']=='assistant'),'')


def inverted_possession(question:str,reply:str)->bool:
    """Catch obvious reversals of positive possession in a conditional question.

    If uncertain, this check must return False rather than overrule the model.
    """
    m=POSSESSION.search(question)
    if not m or not NEGATED_PREMISE.search(reply):return False
    subject_tokens=normalized_tokens(m.group(1))
    # Check same tangible noun appears near the opposite/negated premise.
    for hit in NEGATED_PREMISE.finditer(reply):
        region=reply[max(0,hit.start()-15):min(len(reply),hit.end()+75)]
        if subject_tokens & normalized_tokens(region):return True
    return False


def violates_exclusion(question:str,reply:str,memory:dict)->bool:
    """Check concrete exclusions against imperative advice, not neutral mentions."""
    idx=pick_topic(question,memory or fresh_memory())
    old_user = ([t['text'] for t in memory['topics'][idx]['turns'] if t['role']=='user'][-5:]
                if idx is not None else [])
    for text in [*old_user,question]:
        for excluded in EXCLUSION.finditer(text):
            prohibited=normalized_tokens(excluded.group(1))
            if len(prohibited)<1:continue
            for advice in ADVICE.finditer(reply):
                recommended=normalized_tokens(advice.group(1))
                if len(prohibited & recommended)>=max(1,min(2,len(prohibited))):
                    return True
    return False


def role_drift(question:str,reply:str)->bool:
    return bool(SENIOR.search(question) and ROLE_DRIFT.search(reply) and not re.search(r'(?i)\bconsultant\b|\badvisor\b|مستشار',question))


def identical_rejection(question:str,reply:str,memory:dict)->bool:
    if not CORRECTION.search(question):return False
    old=previous_answer(question,memory)
    if len(old)<40 or len(reply)<40:return False
    clean=lambda x:re.sub(r'\s+',' ',re.sub(r'^\s*\d+[.)]\s*','',x,flags=re.M)).lower().strip()
    return difflib.SequenceMatcher(None,clean(old),clean(reply)).ratio()>=.80


def is_cutoff(reply:str)->bool:
    r=reply.strip()
    if not r:return True
    if r.endswith((':','،',',',';', '-', '—', '**','...')):return True
    if re.search(r'(?i)(?:\b(?:and|or|to|for|because|if|with|the|a|an)\s*$|(?:و|أو|لأن|لكن|حاول|مثلاً|من\s+أجل)\s*$)',r):return True
    if re.search(r'(?m)^\s*\d+[.)]\s*$',r):return True
    # Open markdown list, unfinished bold, dangling colons / connectors.
    if r.count('**') % 2: return True
    return False


def goal_shift_on_duration(question: str, reply: str, memory: dict) -> bool:
    """Narrow detector: follow-up duration to a prior goal vs only duration to a side activity.

    This catches a known failure pattern but is not a general semantic verifier.
    """
    if not wants_time(question): return False
    idx=pick_topic(question, memory or fresh_memory())
    if idx is None:return False
    first=next((t['text'] for t in memory['topics'][idx]['turns'] if t['role']=='user'), '')
    if not first:return False
    if not (SENIOR.search(first) or re.search(r'(?i)\bcareer\b|\bjob\b|وظيفة|مهنة|مستوى\s+وظيفي',first)):
        return False
    if SENIOR.search(reply) or re.search(r'(?i)وظيفة|مستوى\s+وظيفي|\bcareer\b|\bjob\b',reply):
        return False
    if not DURATION.search(reply):return False
    # Suspicious when the duration is ONLY for building projects, a different task.
    return bool(re.search(r'(?i)إعداد\s+المشاريع|تجهيز\s+المشاريع|إنجاز\s+المشاريع|\b(?:finish|build|create)\s+(?:the\s+)?projects?\b',reply))


def suspiciously_short_senior_timeline(question: str, reply: str, memory: dict)->bool:
    """Sanity check, NOT an authoritative seniority benchmark.

    If a question is about reaching a senior professional role and the response
    ONLY offers months/1 year, this usually measures training/projects, not the
    experience needed for a senior job. Requires the user not to have mentioned
    long professional experience; never override a genuinely experienced user.
    """
    if not wants_time(question): return False
    idx = pick_topic(question, memory or fresh_memory())
    prev = [] if idx is None else [t['text'] for t in memory['topics'][idx]['turns'] if t['role']=='user']
    user = ' '.join(prev[-6:] + [question])
    if not SENIOR.search(user): return False
    # Do not flag timelines conditioned explicitly on 3+ years of prior employment.
    if re.search(r'(?i)(?:[3-9]|1\d)\s*(?:years?|سنوات?|سنين|سنه|سنة).{0,35}(?:experience|خبرة|شغل|work)',user):
        return False
    durations = list(DURATION.finditer(reply))
    if not durations: return False
    years=[]
    for m in durations:
        phrase=m.group(0)
        nums=re.findall(r'\d+',phrase)
        if not nums: continue
        num=max(map(int,nums))
        if re.search(r'(?i)سنة|سنوات|سنين|سنه|years?|عام|أعوام',phrase):
            years.append(num)
        elif re.search(r'(?i)شهر|أشهر|شهور|months?',phrase):
            years.append(num/12)
        elif re.search(r'(?i)week|يوم|أسبوع',phrase):
            years.append(0)
    return bool(years and max(years) <= 2.0)


def evaluate_reply(question:str,reply:str,memory:dict)->List[str]:
    """Conservative checks: detect visible failures, not a factual accuracy guarantee."""
    issues=[]
    if role_drift(question,reply):issues.append('role_drift')
    if inverted_possession(question,reply):issues.append('inverted_condition')
    if violates_exclusion(question,reply,memory):issues.append('violated_user_exclusion')
    if wants_time(question) and (not DURATION.search(reply) or SLASH_DURATION.search(reply)):
        issues.append('missing_or_ambiguous_time')
    if REFUSAL.search(reply) and BENIGN.search(question) and not RISKY.search(question):issues.append('unwarranted_refusal')
    if EVAL_REQUEST.search(question) and len(reply)>130 and not re.match(r'(?i)^\s*(?:yes|no|partly|not\s+(?:exactly|quite)|نعم|لا|أيوه|ايوه|جزئيا|جزئي|مش\s+بالضبط)',reply):
        issues.append('missed_evaluation')
    if identical_rejection(question,reply,memory):issues.append('repeated_answer')
    if goal_shift_on_duration(question,reply,memory): issues.append('wrong_time_target')
    if is_cutoff(reply):issues.append('cutoff')
    if re.search(r'(?im)^\s*(?:\*\*(?:السؤال|الإجابة المطلوبة)\*\*|✅\s*(?:ACTION|TIME)|MANDATORY parts of the LAST question|USER PREMISE quoted EXACTLY)', reply):
        issues.append('internal_checklist_leak')
    if suspiciously_short_senior_timeline(question, reply, memory):
        issues.append('implausible_seniority_time')
    return issues


def repair_messages(original:list,question:str,first_reply:str,issues:List[str])->List[dict]:
    descriptions={
        'role_drift':'You replaced a job SENIORITY level with a consultant role. Keep the job level the user named.',
        'inverted_condition':'You reversed the user\'s IF/HAVE premise. Keep the original polarity and rewrite your answer for THAT scenario.',
        'violated_user_exclusion':'You recommended an action the user explicitly excluded. Stop recommending it, and give alternatives consistent with the user preferences.',
        'missing_or_ambiguous_time':'The user asked HOW LONG. Give an honest conditional timeframe with numerical units, not list numbers or a slash of incompatible units. If personal data is missing, explain assumptions.',
        'unwarranted_refusal':'The request seems benign. If it is safe, respond normally; do not invent a refusal.',
        'missed_evaluation':'You were asked to evaluate your own previous advice. Answer yes/no/partly and assess it before giving new advice.',
        'repeated_answer':'The user rejected your earlier advice; address their correction directly and do not repeat the same list.',
        'cutoff':'Your answer ended mid-sentence or mid-list. Give a complete, coherent answer.',
        'internal_checklist_leak':'You revealed internal checklist or task framing. Replace it with a natural direct answer, not a copy of the rubric.',
        'wrong_time_target':'Your answer measured a side activity, not the user\'s ORIGINAL target. Answer the timeline to their target in context, accounting for their projects only as a factor.',
        'implausible_seniority_time':'You stated that a senior professional engineering role can be reached in months without evidence of previous production experience. Distinguish reaching employability from senior-level ownership in production, give a realistic conditional multi-year estimate and do not invent the user work history.',
    }
    changes='\n'.join('- '+descriptions[k] for k in issues)
    return original + [
        {'role':'assistant','content':first_reply[:900]},
        {'role':'user','content':(
            'Please REPLACE your previous draft. It failed the following quality checks:\n'+changes+
            '\nOriginal question (verbatim): '+question[:650]+
            '\nAnswer it fully and directly without making up facts or treating old assistant advice as truth.')}
    ]


def choose_budget(question:str)->int:
    # Avoid cutting multi-part / long answers, but don't force 400-token boilerplate.
    if len(requested_aspects(question))>=3:return 520
    if HOW_REQUEST.search(question) or wants_time(question):return 460
    if re.search(r'(?i)\b(?:explain|steps|detailed|code|plan)\b|بالتفصيل|كود|اشرح|خطة',question):return 330
    return 320


def likely_benign_refusal(reply:str,question:str)->bool:
    return 'unwarranted_refusal' in evaluate_reply(question,reply,fresh_memory())


def guarded_stream(question:str,memory:dict,model_stream:Callable[[list,int],Iterator[str]])->Iterator[str]:
    """Stream a draft, then only regenerate once on an observable failure.

    model_stream(messages, budget) yields cumulative answer strings. Model quality
    cannot be certified by these checks; they catch a limited set of failures.
    """
    original=build_messages(question,memory)
    text=''
    for text in model_stream(original,choose_budget(question)):
        yield text
    if not text.strip():
        raise RuntimeError('The model produced an empty answer.')
    issues=evaluate_reply(question,text,memory)
    if not issues:
        return
    print('V10 quality checks:',', '.join(issues),'— retrying once')
    revised=''
    try:
        for revised in model_stream(repair_messages(original,question,text,issues),choose_budget(question)):
            pass  # keep the first streaming text on screen until a reviewed revision is ready
    except Exception as exc:
        print('V10 recheck error:',type(exc).__name__,str(exc)[:150])
    if revised and not evaluate_reply(question,revised,memory):
        yield revised.strip()
        return
    if 'inverted_condition' in issues or 'role_drift' in issues or 'violated_user_exclusion' in issues or 'wrong_time_target' in issues or 'implausible_seniority_time' in issues:
        # Do not append an irrelevant career answer when the original topic is different.
        idx=pick_topic(question,memory or fresh_memory())
        start= '' if idx is None else next((t['text'] for t in memory['topics'][idx]['turns'] if t['role']=='user'), '')
        if SENIOR.search(question+' '+start):
            yield ('الرد السابق لم يقدّر زمن الوصول إلى مستوى Senior بدقة كافية. المشاريع القوية تسرّع تأهيلك للعمل، '
                   'لكن المسمى الوظيفي Senior عادة يحتاج سنوات من المسؤولية عن أنظمة منشورة والعمل مع فرق. '
                   'مش ممكن أحسب عدد شهور موثوق من غير معرفة خبرتك المهنية الفعلية.')
        else:
            yield 'الرد السابق خلط بين شروط سؤالك أو خالف معلومة صححتها. مش هعتبره إجابة موثوقة؛ وضّح النقطة الأساسية عشان أجاوبك بدقة.'
    elif 'unwarranted_refusal' in issues:
        yield 'حصل خطأ في فهم طلبك العادي. ممكن تعيد صياغة السؤال مرة تانية؟'
    elif 'missing_or_ambiguous_time' in issues:
        yield (text.rstrip()+'\n\n**بالنسبة للمدة:** مش هقدر أحدد رقم موثوق '
               'من غير معرفة نقطة البداية ومستوى الخبرة المطلوبة. '
               'لو وضّحتهم، أقدر أقدّم تقديرًا مشروطًا.')
    elif 'repeated_answer' in issues or 'missed_evaluation' in issues or 'cutoff' in issues or 'internal_checklist_leak' in issues:
        yield 'واضح إن الرد السابق ما استجابش كويس للتصحيح أو ما اكتملش. ممكن تعيد سؤالك بصيغة أدق؟'
    else:
        yield text.rstrip()
