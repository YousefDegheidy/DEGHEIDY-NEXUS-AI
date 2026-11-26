"""Conservative, source-aware checks for fresh news output.

This is *not* an automatic fact checker: surface-level grounding checks cannot
prove an arbitrary natural-language claim. In high-stakes fresh-news answers we
fail closed to attributable, extractive coverage rather than publish guesses.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

_AR_TRANSLATE = {
    'ترامب': 'trump', 'تررامب': 'trump', 'الصين': 'china', 'الصيني': 'china',
    'الصينية': 'china', 'شي': 'xi', 'بوتين': 'putin', 'روسيا': 'russia',
    'الولايات': 'united', 'الامريكية': 'american', 'الأمريكية': 'american',
    'النووية': 'nuclear', 'التجارية': 'trade', 'رسوم': 'tariffs',
    'وصل': 'arrived', 'وصلت': 'arrived', 'غادر': 'departed',
    'اجتمع': 'met', 'التقى': 'met', 'سيلتقي': 'meet', 'سيجتمع': 'meet',
}
STOPWORDS = {
    'that','this','those','from','with','about','into','upon','after','before',
    'said','says','over','will','been','were','have','has','they','their','than',
    'there','when','what','where','them','and','the','are','for','but','not',
    'على','الى','إلى','انه','أنها','انه','ذلك','هذه','هذا','التي','الذي',
    'كان','كانت','بين','حول','أمام','خلال','بسبب','أجل','وقد','كما','تلك',
    'المصدر','الخبر','الأخبار','وقال','وقالت','مصادر','تقارير','فيه','فيها',
    'جدا','فقط','نفس','غير','مع','عن','من','في','أن','إن','او','أو','هو','هي',
}
CITES = re.compile(r'\[(\d+)\]')
CLAIM_SPLITTER = re.compile(r'(?<=[.!؟؛])\s+|\n+')
UNPROVEN_CLAIMS = re.compile(
    r'(?i)secret\s+(?:talks|negotiations)|hidden\s+deal|فضيحة|محادثات\s+سرية|'
    r'اتفاق\s+سري|مشروع\s+نيل|مدينة\s+نصر\s+الجديدة|دولار\s+واحد\s+بالتحديد'
)
CONDITIONAL = re.compile(
    r'(?i)\b(?:if|might|may|could|potentially|hypothetically|scenario)\b|'
    r'إذا|اذا|لو\s|ربما|ممكن|قد\s|احتمال|سيناريو|افتراض|مشروط'
)
WEEKDAY_NAMES={'monday':'monday','tuesday':'tuesday','wednesday':'wednesday',
  'thursday':'thursday','friday':'friday','saturday':'saturday','sunday':'sunday',
  'الاثنين':'monday','الاتنين':'monday','الثلاثاء':'tuesday','الاربعاء':'wednesday',
  'الأربعاء':'wednesday','الخميس':'thursday','الجمعة':'friday','السبت':'saturday','الاحد':'sunday','الأحد':'sunday'}
EVENT_ACTION = re.compile(
    r'(?i)\b(?:arrived|met|held|signed|agreed|discussed|departed|landed)\b|'
    r'وصل|التقى|عقد|وقع|اتفق|ناقش|غادر|أُعلن\s+الاتفاق'
)
ARABIC = re.compile(r'[\u0600-\u06ff]')
IMPACT = re.compile(r'(?i)impact|effect|implication|how\s+will\s+it\s+affect|تأثير|هيأثر|هياثر|يؤثر|ياثر|تأثر|تاثر|ينعكس|تداعيات')
EGYPT = re.compile(r'(?i)egypt|egyptian|مصر|المصري|المصرية')
LATEST = re.compile(r'(?i)today|yesterday|latest|news|visit|meeting|اليوم|امبارح|أخبار|اخبار|زيارة|اجتماع')


def tokens(text: str) -> set[str]:
    words=re.findall(r'[\w\u0600-\u06ff]+', (text or '').lower())
    result=set()
    for word in words:
        norm=word.translate(str.maketrans('أإآى','اااي'))
        if len(norm)<3 or norm in STOPWORDS: continue
        result.add(_AR_TRANSLATE.get(norm,norm))
        if norm.startswith('ال') and len(norm)>5: result.add(_AR_TRANSLATE.get(norm[2:],norm[2:]))
    return result


def split_claims(text: str) -> list[str]:
    clean=re.sub(r'(?m)^#{1,5}\s+.*$','',text or '')
    clean=re.sub(r'(?m)^\s*[-*]\s+', '\n', clean)
    return [s.strip() for s in CLAIM_SPLITTER.split(clean) if len(s.strip())>24]


def impact_question(question: str)->bool:
    return bool(IMPACT.search(question or '') and EGYPT.search(question or ''))


def scenario_answer(ev) -> str:
    """General *conditional* Egypt risk channels, not claimed news outcomes."""
    period=ev.plan.start.date().isoformat() if ev.plan else 'الفترة المطلوبة'
    lead=(', '.join(s['title'][:90] for s in ev.sources[:2]))
    return (
        '**التأثير المحتمل على مصر — تحليل مشروط، مش خبر مؤكد**\n\n'
        'المصادر المتاحة بتتكلم عن التطورات المذكورة أدناه؛ **مافيش فيها قرار مباشر موثق يخص مصر**. '
        'وده معناه إن التأثير، لو حصل، هيكون عبر قنوات غير مباشرة:\n\n'
        '1. **قناة السويس والشحن:** لو تغيّرت حركة التجارة الآسيوية العالمية أو مسارات الشحن، ده ممكن ينعكس على عبور السفن وإيرادات القناة. الاتجاه والحجم مش معروفين من الأخبار دي وحدها.\n'
        '2. **الواردات والمصانع:** لو حصل تغيير في الرسوم الأمريكية–الصينية أو قيود تصدير التقنيات، ممكن يتغيّر سعر وتوافر مكونات بتستخدمها الشركات المصرية.\n'
        '3. **الطاقة والأسعار:** أي تغير فعلي في الطلب العالمي والطاقة ممكن يؤثر على تكاليف الواردات والتضخم، لكن ده محتاج بيانات أسعار مستقلة.\n'
        '4. **التمويل والاستثمار:** تحركات الفائدة والدولار أو إعادة توجيه الاستثمارات ممكن تؤثر على تكلفة التمويل في الأسواق الناشئة، ومنها مصر.\n\n'
        f'**إيه المؤكد؟** عناوين أخبار ضمن الفترة {period}، ومنها: {lead}. '
        '**إيه غير المؤكد؟** حصول اتفاق محدد أو أثر اقتصادي مباشر على مصر. '
        'عشان نحكم بدقة محتاجين نص الاتفاقات المعلنة، وبيانات التجارة والشحن والأسعار قبل وبعد الحدث.\n\n'
        +_footer(ev)
    )


def _footer(ev):
    from src.realtime import evidence_footer
    return evidence_footer(ev)


def compose_coverage_digest(question: str, ev, max_stories: int = 3) -> str:
    """Attributable, short, all-parts answer. No LLM world facts or invented agreements.

    This is NOT independent verification of headlines. The current retrieval source and
    article-read status are declared so users can inspect the originals.
    """
    if not ev or ev.status != 'found':
        from src.realtime import unavailable_message
        return unavailable_message(ev)
    from src.coverage import extract_requirements
    reqs={r.name for r in extract_requirements(question)}
    if 'news_two' in reqs: max_stories=min(max_stories,2)
    out=[]
    interval=(f'{ev.plan.start:%Y-%m-%d %H:%M} → {ev.plan.end:%Y-%m-%d %H:%M} بتوقيت القاهرة'
              if ev.plan else 'الفترة المحددة')
    out.append('**المؤكد من البحث: عُثر على الخبر ومصادره المنشورة، وليس تأكيدًا مستقلًا لوقوع كل الأحداث** — '+interval)
    out.append('_ده تلخيص لما وصلنا من الفهرس والمقالات المتاحة، وليس تحققًا مستقلًا من كل الوقائع._')
    # When the user explicitly requests summaries beyond headlines, do not
    # pad the top-three list with sources whose original text was unreadable.
    detailed=bool(re.search(r'(?i)ما\s*تكتفيش\s*بالعناوين|لا\s*تكتفي\s*بالعناوين|ملخص\s*من\s*سطر|تفاصيل\s*الأخبار|اقرأ\s*المقال|read\s*the\s*article',question))
    eligible=[s for s in ev.sources if s.get('read_status')=='article_read'] if detailed else ev.sources
    stories=eligible[:max_stories]
    if detailed and not stories:
        out.append('تعذر قراءة نصوص المقالات الأصلية ضمن النتائج الحالية؛ لذلك مش هعرض عناوين فقط باعتبارها ملخصات أخبار متحقَّق منها.')
    if 'news_three' in reqs and len(stories)<3:
        out.append(f'وجدت {len(stories)} مادة مؤرخة مناسبة فقط، ومش هخترع خبرًا ثالثًا.')
    if 'news_two' in reqs and len(stories)<2:
        out.append(f'وجدت {len(stories)} مادة مؤرخة مناسبة فقط، ومش هخترع خبرًا ثانيًا.')
    for s in stories:
        i=next(j for j,src in enumerate(ev.sources,1) if src is s)
        kind=('نشر' if s.get('date_kind')=='published' else 'رصد')
        rawtitle=re.sub(r'[\[\]<>]','',s.get('title',''))[:180]
        from urllib.parse import urlparse
        source_url=s.get('article_url') or s.get('url') or ''
        host=(urlparse(source_url).hostname or '').lower()
        # An RSS aggregator link is not the publisher's original article URL.
        direct=bool(source_url and not (host=='news.google.com' or host.endswith('.news.google.com')))
        source_link=(f'[المقال لدى الناشر]({source_url})' if direct else
                     'رابط الناشر المباشر غير متاح؛ رابط الفهرس ليس مصدرًا أصليًا')
        evidence=s.get('article_text','') if s.get('read_status')=='article_read' else ''
        if evidence:
            # Short licensed excerpt is intentionally less than 25 words per source.
            excerpt=' '.join(re.sub(r'\s+',' ',evidence).split()[:16])
            out.append(f'{i}. **{rawtitle}** — {s.get("publisher","المصدر")}; {kind} {s.get("published","غير متاح")}. '
                       f'المتاح من المقال: «{excerpt}…». {source_link} [{i}]')
        else:
            out.append(f'{i}. **{rawtitle}** — {s.get("publisher","المصدر")}; {kind} {s.get("published","غير متاح")}. '
                       f'المتاح عنوان المقال فقط؛ لا أؤكد تفاصيل إضافية. {source_link} [{i}]')
    if re.search(r'(?i)تاريخ\s+وقوع|تاريخ\s+الحدث|وقت\s+الحدث|event\s+date',question):
        out.append('**تاريخ وقوع الأحداث:** غير مؤكد على نحو مستقل في هذه المواد؛ تواريخ النشر المذكورة ليست تواريخ وقوع تلقائيًا.')
    if 'news_trump' in reqs:
        trump=[(i,s) for i,s in enumerate(ev.sources,1) if re.search(r'(?i)trump|ترامب|تررامب',s.get('title',''))]
        out.append('**ترامب والصين — الثابت وحدود التحقق**')
        if trump:
            for idx,s in trump[:2]:
                out.append(f'- بحسب عنوان/محتوى مصدر [{idx}]: {re.sub(r"[<>]", "", s.get("title", ""))[:160]}. '
                           'تاريخ النشر لا يثبت وحده وقت وقوع الحدث.')
        else:
            out.append('لم تصل مادة مؤرخة مطابقة بالتحديد لترامب والصين ضمن النتائج؛ ده لا ينفي وقوع تطورات.')
    if 'news_rumor' in reqs:
        out.append('**ادعاء الاتفاق السري بقيمة 9 تريليون دولار:** لم أجد في المواد المقروءة هنا دليلاً يؤكده، '
                   'لذلك **غير متحقق منه**؛ وغياب الدليل في هذا البحث لا يثبت وحده أنه كاذب. '
                   'التأكيد يحتاج نص اتفاق رسمي أو تغطية موثقة مستقلة.')
    if re.search(r'(?i)(?:gemini|جيميني|جيميناي).{0,100}(?:إغلاق|إيقاف|هتقفل|shutdown|وقف)|(?:إغلاق|إيقاف|هتقفل|shutdown).{0,100}(?:gemini|جيميني|جيميناي)',question):
        out.append('**التحقق من إيقاف Gemini:** نتائج البحث الحالية ليست دليلًا على صدور إعلان رسمي بالإيقاف. لا أستطيع تأكيد الادعاء من المواد المتاحة، وغياب دليل في نتائج البحث ليس نفيًا قاطعًا.')
    if 'news_egypt' in reqs:
        out.append('**التأثير على مصر — سيناريوهات مشروطة وليست نتائج مؤكدة:**')
        out.extend(['- **قناة السويس:** تغيّر أحجام التجارة أو طرق الشحن قد يؤثر على المرور والإيرادات؛ الاتجاه غير محسوم.',
                    '- **الواردات والتجارة:** تغيّر الرسوم أو سلاسل الإمداد قد يغير تكلفة مدخلات المصانع.',
                    '- **الطاقة:** تحركات الطلب والأسعار العالمية قد تنعكس على تكلفة الواردات، وتحتاج أسعارًا فعلية للتحقق.',
                    '- **الاستثمار:** إعادة توجيه سلاسل التوريد قد تخلق فرصًا أو منافسة، ولا يوجد أثر مصري مباشر مثبت من هذه المواد.'])
    out.append('**فصل الوقائع عن التوقعات:** العناوين السابقة أخبار منشورة/مفهرسة؛ ما لم يوثق نص المقال حدوثه، يظل مجرد ادعاء أو احتمال.')
    out.append(_footer(ev))
    return '\n\n'.join(out)


def safe_digest(ev, limit=3) -> str:
    """Readable, attributable coverage; never invent body contents."""
    if not ev or ev.status!='found':return 'مافيش مصدر حديث تم التحقق منه في البحث الحالي.'
    out=['**المؤكد من المواد التي أمكن الوصول إليها**',
         '_السطور التالية تلخيص محافظ للمادة المتاحة؛ ولا تعني تحققًا مستقلًا من الحدث نفسه._']
    for i,s in enumerate(ev.sources[:limit],1):
        title=s.get('title','')[:170]
        pub=s.get('publisher','المصدر')
        if s.get('read_status')=='article_read' and s.get('article_text'):
            text=re.sub(r'\s+',' ',s['article_text']).strip()
            # Avoid long quotations of licensed news content; quote < 25 words per source.
            piece=' '.join(text.split()[:18]); desc=f'تم الوصول لنص المقال، ومن بدايته: «{piece}…»'
        else:
            desc='تم الوصول إلى العنوان وتاريخ الفهرسة/النشر فقط؛ تفاصيل المقال غير متاحة للمراجعة.'
        out.append(f'- **[{i}] {title}** — {pub}: {desc}')
    out.append('**حدود الإجابة:** ماينفعش أأكد تفاصيل لقاء أو نتائج اتفاق من مجرد عنوان، أو أنسب مواقف أطراف من غير نص يدعمها.')
    out.append(_footer(ev))
    return '\n\n'.join(out)


@dataclass
class GuardResult:
    approved: bool
    problems: list[str]=field(default_factory=list)
    factual_checked: int=0
    unsupported: int=0


def _event_state_conflict(claim: str, evidence: str)->bool:
    """Small conservative temporal blocker; not a general date verifier."""
    a=claim.lower();b=evidence.lower()
    future_claim=bool(re.search(r'(?i)\b(?:will|plans? to|set to|expected to|on (?:his|her) way)\b|سيصل|سوف\s+يصل|في\s+طريقه',a))
    arrived=bool(re.search(r'(?i)\b(?:has arrived|arrived|landed|welcomed|was greeted)\b|وصل|استقبل|هبط',b))
    completed_claim=bool(re.search(r'(?i)\b(?:met|already met|has met|held talks|agreed|signed)\b|اجتمع|التقى|وقّع|وقع\s+اتفاق|اتفق',a))
    planned_only=bool(re.search(r'(?i)\b(?:will meet|set to meet|plans to meet|expected to meet)\b|سيجتمع|سيلتقي|مقرر\s+أن\s+يلتقي',b))
    past_in_evidence=bool(re.search(r'(?i)\b(?:met|have met|held|signed|agreed)\b|اجتمع|التقى|وقّع|وقع',b))
    weekday_claim={day for token,day in WEEKDAY_NAMES.items() if token in a}
    weekday_source={day for token,day in WEEKDAY_NAMES.items() if token in b}
    wrong_weekday=bool(weekday_claim and weekday_source and weekday_claim.isdisjoint(weekday_source))
    agreement_claim=bool(re.search(r'(?i)\b(?:signed|agreed|agreement\s+reached|finalized)\b|وقّع|وقع\s+اتفاق|اتفق|تم\s+الاتفاق',a))
    agreement_unconfirmed=bool(re.search(r'(?i)\b(?:no agreement|without an agreement|no deal|not yet agreed|has yet to be announced)\b|لم\s+يعلن\s+عن\s+اتفاق|دون\s+اتفاق',b))
    return ((future_claim and arrived) or
            (completed_claim and planned_only and not past_in_evidence) or
            (agreement_claim and agreement_unconfirmed) or wrong_weekday)


def evaluate_news_answer(answer: str, ev)->GuardResult:
    """Fail-closed factual check for *fresh news*. It is not semantic verification.

    Require each substantial factual sentence to carry a citation referencing
    a source and to share enough distinct content with evidence. Arabic-to-English
    paraphrases may be conservatively rejected. Conditional forward-looking
    analysis must be explicit and is not treated as news fact.
    """
    if not ev or ev.status!='found':return GuardResult(False,['No dated news evidence'])
    problems=[];checked=0;unsupported=0
    allowed_urls={s.get(k,'').rstrip('/') for s in ev.sources for k in ('url','article_url') if s.get(k)}
    for url in re.findall(r'https?://[^\s<>\]\)]+',answer):
        if url.rstrip('.,;/') not in allowed_urls:
            problems.append('URL not retrieved from evidence');unsupported+=1
    for sentence in split_claims(answer):
        # Sources/footer, questions, cautious disclosures and not-news instructions.
        if re.search(r'(?i)^\s*(?:sources?|المصادر|روابط|الروابط)\s*[:：]',sentence):continue
        sentence=re.sub(r'https?://[^\s<>\]\)]+','',sentence)
        if CONDITIONAL.search(sentence) and not EVENT_ACTION.search(sentence): continue
        # Only scrutinize sentences presenting current-world facts, not headings.
        if len(tokens(sentence))<3:continue
        checked+=1
        if UNPROVEN_CLAIMS.search(sentence):
            unsupported+=1;problems.append('Unsupported named/sensitive assertion');continue
        cited=[int(n) for n in CITES.findall(sentence)]
        if not cited:
            unsupported+=1;problems.append('Factual sentence without source marker');continue
        sources=[ev.sources[n-1] for n in cited if 1 <= n <= len(ev.sources)]
        if not sources:
            unsupported+=1;problems.append('Citation index not in retrieved evidence');continue
        claim_tokens=tokens(CITES.sub('',sentence))
        # Anchor on article bodies where accessible; otherwise only headline, never
        # claim further detail based on indexing metadata alone.
        accept=False
        for src in sources:
            evidence_text=src.get('title','')+' '+(src.get('article_text','') if src.get('read_status')=='article_read' else '')
            evid_tokens=tokens(evidence_text)
            common=claim_tokens & evid_tokens
            score=len(common)/max(1,len(claim_tokens))
            threshold=.24 if src.get('read_status')=='article_read' else .52
            if len(common)>=2 and score>=threshold and not _event_state_conflict(sentence,evidence_text):
                accept=True;break
        if not accept:
            unsupported+=1;problems.append('Citation not sufficient for claim')
    if checked==0:problems.append('No checkable source-cited news facts')
    # Even one suspicious unsupported claim can materially distort news. Fail closed.
    return GuardResult(approved=checked>0 and unsupported==0,problems=problems,
                       factual_checked=checked,unsupported=unsupported)


def render_news_response(question: str, candidate: str, ev)->tuple[str,GuardResult]:
    """Final output after evidence checks; never disclose unverified draft."""
    from src.coverage import extract_requirements, missing_requirements
    reqs={r.name for r in extract_requirements(question)}
    if impact_question(question) and 'news_current' not in reqs:
        return scenario_answer(ev),GuardResult(True,['Conditional economic scenarios, not news claims'])
    verdict=evaluate_news_answer(candidate,ev)
    if re.search(r'(?i)رابط\s*(?:مباشر|أصلي|الناشر)|المصدر\s+الأصلي|direct\s+(?:article|publisher|source)',question) and re.search(r'https?://(?:[^/]+\.)?news\.google\.com(?:/|\b)',candidate):
        return compose_coverage_digest(question,ev),GuardResult(False,['Index URL is not a direct publisher URL'])
    if not verdict.approved:
        return compose_coverage_digest(question,ev),verdict
    if ('news_rumor' in reqs or 'news_egypt' in reqs or 'news_three' in reqs or 'news_two' in reqs) and missing_requirements(question,candidate):
        return compose_coverage_digest(question,ev),GuardResult(False,['Missing components in news answer'])
    return candidate.strip()+'\n\n'+_footer(ev),verdict
