"""Conservative news selection and official-source claim checking.
Retrieval is not independent verification of real events. Fail closed on unreadable sources.
"""
from __future__ import annotations
import re
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from src.article_reader import readable_article, public_http_url

EXCLUDE_ANALYSIS=re.compile(r'(?i)(?:استبعد|بدون|من\s+غير|لا\s+تضع|exclude|no\s+opinion).{0,90}(?:الرأي|رأي|تحليل|opinion|analysis)')
DIRECT_REQUEST=re.compile(r'(?i)رابط[اًًا]?\s*(?:مباشر|أصلي|الناشر)|المصدر\s+الأصلي|المقال\s+الأصلي|direct\s+(?:article|publisher|source|link)')
ANALYSIS_TITLE=re.compile(r'(?i)^(?:what\s+(?:did|does|will|can|is)|why\b|how\s+(?:will|can|did|does)|what\s+it\s+means|the\s+case\s+for|explainer\s*:|analysis\s*:|رأي\s*[:：]|تحليل\s*[:：]|ماذا\s+يعني|لماذا\s+|كيف\s+نفهم|قراءة\s+في)|\b(?:opinion|commentary|op-ed|editorial|analysis|explainer|thinkpiece)\b|الخبير\s+السياسي|وجهة\s+نظر')
ANALYSIS_PATH=re.compile(r'(?i)/(?:opinion|opinions|commentary|editorials?|analysis|perspectives?|voices|columns?|op-ed|ideas)(?:/|$)')
# Explicitly analysis-oriented publications; not a universal publisher blacklist.
ANALYSIS_PUBLISHER=re.compile(r'(?i)\b(?:foreign policy|foreign affairs|american enterprise institute|us.china business council|council on foreign relations)\b')


def analytical(source):
    title=source.get('title','');pub=source.get('publisher','');url=source.get('article_url') or source.get('url','')
    # When explicitly asking for factual headlines, uncertain question-led items are excluded.
    meta=' '.join(str(source.get(k,'')) for k in ('article_type','section','genre'))
    framing=re.search(r'(?i)^(?:the\s+(?:limits|future|challenge|dilemma|cost|meaning|lessons)\s+of|what\s+to\s+expect|لماذا|إلى\s+أين|مستقبل\s+)',title.strip())
    return bool(framing or ANALYSIS_TITLE.search(title) or ANALYSIS_PATH.search(urlparse(url).path) or ANALYSIS_PUBLISHER.search(pub) or re.search(r'(?i)opinion|analysis|commentary|editorial|رأي|تحليل',meta))


def strict_news_request(q):
    return bool(EXCLUDE_ANALYSIS.search(q) and DIRECT_REQUEST.search(q))


def direct_url(s):
    url=s.get('article_url') or s.get('url','')
    if not public_http_url(url):return ''
    host=(urlparse(url).hostname or '').lower()
    return '' if host=='news.google.com' or host.endswith('.news.google.com') else url


def strict_story_digest(question,evidence):
    """Only article-read, direct, non-analysis, dated sources. No LLM invention.
    Extract short text evidence rather than label headline fragments a summary.
    """
    q=question.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩','0123456789'))
    m=re.search(r'(?i)(\d+)\s*(?:أخبار|اخبار|news|headlines)',q)
    count=min(10,max(1,int(m[1]))) if m else (2 if re.search(r'خبرين|خبران|two\s+news',q,re.I) else 3)
    valid=[];seen=set()
    for s in evidence.sources:
        url=direct_url(s)
        if not url or analytical(s) or s.get('date_kind')!='published' or s.get('read_status')!='article_read':continue
        raw=re.sub(r'\s+',' ',s.get('article_text','')).strip()
        # Remove repeated article headline, then select a sentence with substance.
        title=re.sub(r'\s+',' ',s.get('title','')).strip()
        for heading in (title,title.rsplit(' - ',1)[0]):
            if heading and raw.lower().startswith(heading.lower()):raw=raw[len(heading):].lstrip(' :–-')
        if len(raw.split())<12:continue
        two=bool(re.search(r'جملتين|two sentences|2 sentences',question,re.I))
        sentences=[v.strip() for v in re.findall(r'[^.!؟]+[.!؟](?=\s|$)',raw) if len(v.split())>=7]
        if two and len(sentences)<2:continue
        key=re.sub(r'\W','',title.lower())[:100]
        if key in seen:continue
        seen.add(key);valid.append((s,url,raw))
        if len(valid)>=count:break
    lines=[f'**نتائج فحص الأخبار ({evidence.searched_at})** — {len(valid)} من {count} مواد اجتازت شروط الاختيار.']
    if evidence.plan:lines.append(f'نافذة النشر: {evidence.plan.start:%Y-%m-%d %H:%M} → {evidence.plan.end:%Y-%m-%d %H:%M} بتوقيت القاهرة.')
    if len(valid)<count:lines.append('لم أجد العدد المطلوب من التقارير المقروءة ذات الرابط المباشر؛ استبعدت المواد التحليلية أو غير المقروءة، ومش هخترع أخبارًا إضافية.')
    for i,(s,url,raw) in enumerate(valid,1):
        title=re.sub(r'[\[\]<>]','',s['title'])[:200]
        # bounded quotation, explicitly labeled, not an invented paraphrase
        first=re.split(r'(?<=[.!؟])\s+',raw)[0]
        excerpt=' '.join(first.split()[:22]); suffix='…' if len(first.split())>22 else ''
        if re.search(r'جملتين|two sentences|2 sentences',question,re.I):
            sentences=[v.strip() for v in re.findall(r'[^.!؟]+[.!؟](?=\s|$)',raw) if len(v.split())>=7]
            excerpt=' '.join(' '.join(v.split()[:22])+('…' if len(v.split())>22 else '') for v in sentences[:2]);suffix='' 
        lines.append(f'{i}. **{title}** — {s.get("publisher", "المصدر")}\n'
                     f'تاريخ النشر بحسب الفهرس: {s.get("published", "غير متاح")}؛ تاريخ الحدث: غير محدد من الأدلة المستخرجة.\n'
                     f'مقتطف من متن التقرير (وليس تأكيدًا مستقلًا للحدث): «{excerpt}{suffix}»\n'
                     f'[المقال الأصلي لدى الناشر]({url})')
    lines.append('الفهرسة لا تثبت زمن وقوع الحدث. التصنيف محافظ وليس ضمانًا بأن كل مادة تقرير خبري خالص.')
    return '\n\n'.join(lines)


# Domain allowlists constrain official search; unknown organizations fail back to normal evidence routing.
_OFFICIAL=[
    (r'(?i)\bnvidia\b|إنفيديا|انفيديا|\bcuda\b', ('nvidia.com','docs.nvidia.com','developer.nvidia.com'), 'NVIDIA'),
    (r'(?i)\bazure\b', ('learn.microsoft.com','azure.microsoft.com','microsoft.com'), 'Microsoft Azure'),
    (r'(?i)\bcolab\b|كولاب', ('colab.research.google.com','research.google.com','support.google.com','blog.google','developers.google.com'), 'Google Colab'),
    (r'(?i)\bchatgpt\b|\bopenai\b|شات\s*جي\s*بي',('openai.com','help.openai.com','status.openai.com'),'OpenAI ChatGPT'),
    (r'(?i)github\s+copilot|جيت\s*هاب',('github.blog','docs.github.com','support.github.com'),'GitHub'),
    (r'(?i)\bgemini\b|جيمين',('blog.google','support.google.com','gemini.google.com'),'Google Gemini'),
    (r'(?i)\bmicrosoft\b|مايكروسوفت',('microsoft.com','blogs.microsoft.com','support.microsoft.com'),'Microsoft'),
]
_VERIFY=re.compile(r'(?i)تحقق|تأكد|مصادر\s+رسمية|إعلان[اً]?\s+رسمي|اعلان[اً]?\s+رسمي|رسمياً|رسميًا|verify|official|fact.check')
_CLAIM=re.compile(r'(?i)إغلاق|اغلاق|إيقاف|ايقاف|وقف|توقف|هتقفل|إلغاء|shutdown|shut\s*down|discontinu|terminat|acquir|استحواذ|صفقة')


def official_target(question):
    if not (_VERIFY.search(question) and _CLAIM.search(question)):return None
    for pat,domains,name in _OFFICIAL:
        if re.search(pat,question):return domains,name
    return None


def allowed_url(url,domains):
    if not public_http_url(url):return False
    p=urlparse(url);host=(p.hostname or '').lower()
    return p.scheme=='https' and not p.username and not p.password and any(host==d or host.endswith('.'+d) for d in domains)


def _search(q):
    from ddgs import DDGS
    with DDGS(timeout=8) as d:return list(d.text(q,max_results=5))


def _fetch_official(url,domains):
    import requests
    # A direct redirect to an unofficial site cannot become official evidence.
    # Do not follow redirects here: a blocked/moved page is reported as unreadable.
    if not allowed_url(url,domains):return ''
    def fetch(v):
        with requests.get(v,timeout=(3,7),allow_redirects=False,stream=True,headers={'User-Agent':'Mozilla/5.0 DEGHEIDYQA'}) as r:
            if r.status_code!=200 or not allowed_url(r.url,domains) or 'html' not in r.headers.get('content-type','').lower():return ''
            body=bytearray()
            for chunk in r.iter_content(16384):
                if len(body)+len(chunk)>1000000:break
                body.extend(chunk)
            return bytes(body).decode(r.encoding or 'utf-8',errors='replace'),r.url
    return readable_article(url,fetch=fetch)


def check_official_claim(question,*,search=None,fetch=None):
    target=official_target(question)
    if not target:return None
    domains,name=target;search=search or _search;fetch=fetch or _fetch_official
    # Only fixed allowlisted domains; no user-provided URLs get fetched.
    intent='shutdown discontinued service termination official announcement' if not re.search(r'(?i)استحواذ|acquir|صفقة',question) else 'acquisition official announcement'
    tokens=[t for t in ('CUDA','RTX','Azure AI','Azure','Colab','ChatGPT','Copilot','Gemini') if re.search(r'(?i)\b'+re.escape(t)+r'\b',question)]
    subject=' '.join(tokens) or name
    queries=[f'site:{domain} {subject} {intent}' for domain in domains[:4]]
    if 'CUDA' in tokens:queries[-1]=f'site:docs.nvidia.com CUDA RTX support compatibility release notes deprecation'
    if 'Azure' in tokens:queries[-1]=f'site:learn.microsoft.com Azure AI service retirement lifecycle announcements'

    found=[];errors=0;completed=0
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in [pool.submit(search,q) for q in queries]:
            try:found.extend(f.result());completed+=1
            except Exception:errors+=1
    seen=set();links=[]
    def relevance(r):
        text=' '.join(str(r.get(k,'')) for k in ('title','body','href','url')).lower()
        return sum(t.lower() in text for t in tokens)
    found.sort(key=relevance,reverse=True)
    for r in found:
        if tokens and not relevance(r):continue
        u=r.get('href') or r.get('url','')
        if u not in seen and allowed_url(u,domains):
            seen.add(u);links.append((u,r.get('title','صفحة رسمية')))
        if len(links)>=5:break
    def read(pair):
        u,t=pair
        try:body=fetch(u,domains)
        except Exception:body=''
        return u,t,body
    with ThreadPoolExecutor(max_workers=3) as pool: reviewed=list(pool.map(read,links))
    lines=[f'**التحقق الرسمي من الادعاء بخصوص {name}**',f'البحث الموجّه: {completed}/{len(queries)} عمليات بحث اكتملت؛ {len(links)} نتائج من النطاقات الرسمية؛ {sum(bool(b) for _,_,b in reviewed)} صفحات أمكن قراءة محتواها.']
    if errors:lines.append(f'تعذر تنفيذ {errors} عمليات بحث؛ التغطية غير كاملة.')
    for url,title,body in reviewed:
        title=re.sub(r'[\[\]<>]','',str(title))[:120]
        if body:
            sentence=next((s for s in re.split(r'(?<=[.!؟])\s+',body) if _CLAIM.search(s) and (not tokens or any(t.lower() in s.lower() for t in tokens))), '')
            if sentence:
                excerpt=' '.join(sentence.split()[:22]);lines.append(f'- [{title}]({url}) — مقتطف مرتبط بالإيقاف أو الصفقة: «{excerpt}». سياق الإعلان وتاريخه يحتاجان مراجعة مباشرة.')
            else:lines.append(f'- [{title}]({url}) — الصفحة قُرئت؛ لم يظهر في الجزء المستخرج نص صريح مرتبط بالادعاء.')
        else:lines.append(f'- [{title}]({url}) — نتيجة رسمية من الفهرس، لكن تعذرت قراءة نص الصفحة؛ ليست دليلًا على صحة الادعاء.')
    if not reviewed:lines.append('لم أجد صفحة رسمية مطابقة يمكن عرضها كدليل.')
    lines.append('**الحكم:** الادعاء غير مُثبت من الأدلة التي أمكن فحصها هنا. لا أستنتج نفيًا قاطعًا أو تأكيدًا من نتائج البحث وحدها؛ أي مقتطف يحتاج مطابقة نص الإعلان مع الخدمة والتاريخ المذكورين.')
    return '\n\n'.join(lines)
