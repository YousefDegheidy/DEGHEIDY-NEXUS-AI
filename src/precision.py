"""Deterministic answers for a small set of verifiable, recurring tasks.

These functions deliberately do NOT pretend to be a general calculator or a Python
execution sandbox. They answer only questions whose structures can be parsed safely.
"""
from __future__ import annotations
import re

SEATS = re.compile(r'(?<!\d)(\d{1,5})\s*(?:كرسي|كراسي|مقعد|مقاعد|seats?|places?|capacity\b)', re.I)
PASSENGERS = re.compile(r'(?<!\d)(\d{1,5})\s*(?:راكب|ركاب|عامل|عمال|passengers?|riders?)', re.I)
# Spoken Egyptian Arabic: 'الأول سعته 36 كرسي وراكب فيه 27'.
PASSENGERS_AFTER = re.compile(r'(?:راكب|ركاب|عامل|عمال|passengers?|riders?)\s*(?:فيه|فيها|عددهم|عددها|عدد|هما|هم|=|:)?\s*(?<!\d)(\d{1,5})(?!\d)',re.I)
SEATS_AFTER = re.compile(r'(?:سعته|سعتها|سعة|capacity)\s*(?:=|:)?\s*(?<!\d)(\d{1,5})(?!\d)',re.I)
ORDINAL = re.compile(r'(?<![\w\u0621-\u064a])(?:و)?(?:الأول|الاول|الثاني|التاني|الثالث|التالت|الرابع|الرابعة|الخامس|first|second|third|fourth|fifth)(?![\w\u0621-\u064a])',re.I)
BUS_QUESTION = re.compile(r'(?i)إشغال|اشغال|occupancy|نسبة|احسب|calculate|رتب|ترتيب|الإجمالي|المجموع')
DECLARED_BUS_COUNT = re.compile(r'(?<!\d)([2-9]|[٢-٩])\s*(?:أتوبيسات|اتوبيسات|أتوبيس|اتوبيس|حافلات|باصات|buses?)',re.I)
CODE_REQUEST = re.compile(r'(?i)python|pytest|function|دالة|كود|unit tests?|اختبارات|\bml\b|\bor\b|بحوث العمليات')


def occupancy_info(query: str):
    seats = SEATS.search(query or '')
    people = PASSENGERS.search(query or '')
    if not (seats and people):return None
    n,p=int(seats.group(1)),int(people.group(1))
    if n<=0 or p<0:return None
    return {'seats':n,'passengers':p,'rate':100*p/n,'over_capacity':p>n}


def multiple_bus_info(query: str):
    """Parse complete numbered vehicle facts, rejecting missing buses.

    Ordinals after the data, e.g. 'احسب الأول والثاني والثالث', are
    instructions and MUST NOT be interpreted as more bus records.
    """
    q=query or ''
    labels=list(ORDINAL.finditer(q))
    if len(labels)<2:
        return None
    declared=DECLARED_BUS_COUNT.search(q)
    expected=int(declared.group(1).translate(str.maketrans('٠١٢٣٤٥٦٧٨٩','0123456789'))) if declared else None
    buses=[]
    for i,hit in enumerate(labels):
        # Consider only the span between consecutive labels. An instruction that
        # repeats the labels has no capacity/occupancy facts and is ignored.
        chunk=q[hit.end():(labels[i+1].start() if i+1<len(labels) else len(q))]
        chunk=re.split(r'[.;؛\n]|(?:\s+(?:احسب|بعدها|وبعدين|ورتب|رتبهم|ثم)\s+)',chunk,maxsplit=1)[0]
        seats=SEATS.search(chunk) or SEATS_AFTER.search(chunk)
        people=PASSENGERS.search(chunk) or PASSENGERS_AFTER.search(chunk)
        if not seats and not people:
            continue  # 'الأول والثاني والثالث' in a follow-up instruction
        if not seats or not people:
            return None  # real vehicle with incomplete data, never guess
        n,k=int(seats.group(1)),int(people.group(1))
        if n<=0 or k<0:
            return None
        label=re.sub(r'^و(?=ال)', '',hit.group(0)).lower()
        label={'التاني':'الثاني','التالت':'الثالث','الرابعة':'الرابع','الاول':'الأول'}.get(label,label)
        if label in {b['label'].lower() for b in buses}:
            return None  # contradictory/doubled factual record
        buses.append({'label':label,'seats':n,'passengers':k,
                      'rate':100*k/n,'over_capacity':k>n})
    if len(buses)<2 or (expected is not None and len(buses)!=expected):
        return None
    return buses


def _percent(rate:float) -> str:
    return f'{rate:.0f}%' if rate.is_integer() else f'{rate:.2f}%'


def _single_bus_code() -> str:
    return '''

**دالة Python تتحقق من المدخلات:**
```python
def calculate_occupancy(passengers: int, seats: int) -> float:
    if not isinstance(passengers, int) or not isinstance(seats, int):
        raise TypeError("Inputs must be integers")
    if seats <= 0 or passengers < 0 or passengers > seats:
        raise ValueError("Invalid passenger count or bus capacity")
    return round(100.0 * passengers / seats, 2)
```

**3 اختبارات باستخدام pytest:**
```python
import pytest

def test_regular_bus():
    assert calculate_occupancy(28, 35) == 80.0

def test_zero_seats():
    with pytest.raises(ValueError):
        calculate_occupancy(5, 0)

def test_over_capacity():
    with pytest.raises(ValueError):
        calculate_occupancy(36, 35)
```

**ML:** يتنبأ بعدد الركاب على كل خط وفي كل وردية باستخدام بيانات الحضور السابقة واليوم والوردية ومواقع الركوب.

**OR:** يستخدم توقعات الطلب لتحديد عدد الحافلات وتوزيعها على الخطوط وتحسين الجداول وتقليل التكلفة والانتظار تحت قيود السعة ومواعيد الورديات.'''


def transport_additions(query: str):
    """Compute explicit independent financial metrics; never guess missing data."""
    from decimal import Decimal
    q=(query or '').translate(str.maketrans('٠١٢٣٤٥٦٧٨٩٫٬','0123456789.,'))
    amount=r'(?<![\d.,-])(\d+(?:,\d{3})*(?:\.\d+)?)\s*(ألف|الف|آلاف|الاف|مليون|thousand|million|k)?'
    costs=re.findall(amount+r'\s*(?:جنيه|جنيها|EGP)(?!\w)',q,re.I)
    prod=re.findall(amount+r'\s*(?:متر\s+مربع|م[²2]|square\s+met(?:er|re)s?|m[²2])(?!\w)',q,re.I)
    ratio=bool(re.search(r'تكلفة\s+(?:النقل\s+)?(?:لكل|للمتر)|cost\s+per\s+(?:square\s+met|m[²2])',q,re.I))
    saving=bool(re.search(r'توفير|وفر|saving|savings',q,re.I) and re.search(r'خفض|تقليل|تخفيض|reduce|reduction|cut',q,re.I))
    if not (ratio or saving):return []
    def num(pair):
        n,u=pair
        return Decimal(n.replace(',',''))*(1000000 if u.lower() in ('مليون','million') else 1000 if u else 1)
    def fmt(n):return format(n.quantize(Decimal('0.0001')).normalize(),'f')
    c=num(costs[0]) if len(costs)==1 else None
    lines=[]
    if ratio:
        v=num(prod[0]) if len(prod)==1 else None
        if c is None or v is None:lines.append('**تكلفة المتر:** محتاج تكلفة وإنتاج محددين بدون التباس.')
        elif v==0:lines.append('**تكلفة المتر:** غير معرّفة لأن الإنتاج صفر.')
        else:lines.append(f'**تكلفة النقل لكل متر مربع:** {fmt(c)} ÷ {fmt(v)} = **{fmt(c/v)} جنيه/م²**.')
    if saving:
        rates=re.findall(r'(?:خفض|تقليل|تخفيض|reduce|reduction|cut)[^.!؟؛%]{0,75}?(\d+(?:\.\d+)?)\s*(?:%|٪|percent)',q,re.I)
        r=Decimal(rates[0]) if len(rates)==1 else None
        if c is None or r is None or not 0<=r<=100:lines.append('**التوفير:** محتاج تكلفة أصلية ونسبة خفض واحدة بين صفر و100%.')
        else:lines.append(f'**التوفير:** {fmt(c)} × {fmt(r)}% = **{fmt(c*r/100)} جنيه** لنفس فترة التكلفة. التكلفة بعد الخفض: {fmt(c*(100-r)/100)} جنيه.')
    return lines


def occupancy_solution(query: str):
    if not BUS_QUESTION.search(query or ''):return None
    buses=multiple_bus_info(query)
    if buses is not None:
        total_seats=sum(x['seats'] for x in buses)
        total_passengers=sum(x['passengers'] for x in buses)
        lines=['**نسب الإشغال لكل أتوبيس:**']
        for x in buses:
            lines.append(f'- {x["label"]}: {x["passengers"]} ÷ {x["seats"]} × 100 = **{_percent(x["rate"])}**')
        avg=100*total_passengers/total_seats
        lines.append(f'\n**إشغال الأسطول ككل:** {total_passengers} ÷ {total_seats} × 100 = **{_percent(avg)}**. '
                     'حسب إجمالي الركاب/المقاعد، مش متوسط نسب الأتوبيسات.')
        if re.search(r'(?i)رتب|الترتيب|ranking|order|sort|الأقل|الاعلى|الأعلى',query):
            ranks=sorted(buses,key=lambda x:x['rate'])
            lines.append('**من الأقل للأعلى:** '+ ' ← '.join(f'{x["label"]} ({_percent(x["rate"])})' for x in ranks)+'.')
        if any(b['over_capacity'] for b in buses):
            lines.append('**تنبيه:** فيه عدد ركاب أكبر من المقاعد في أتوبيس واحد على الأقل؛ راجع السلامة.')
        if CODE_REQUEST.search(query):
            lines.append(_single_bus_code())
        lines.extend(transport_additions(query))
        return '\n'.join(lines)
    # A multi-bus question MUST NOT be reduced to its first pair.
    if DECLARED_BUS_COUNT.search(query) or re.search(r'(?i)أتوبيسات|اتوبيسات|buses', query):
        return None
    info=occupancy_info(query)
    if info is None:return '\n'.join(transport_additions(query)) or None
    n,p,rate=info['seats'],info['passengers'],info['rate']
    suffix='\n\n**تنبيه:** عدد الركاب أكبر من عدد المقاعد؛ يلزم مراجعة السلامة.' if p>n else ''
    base=f'**نسبة الإشغال الصحيحة: {_percent(rate)}** = {p}/{n} × 100.'+suffix
    return base+(_single_bus_code() if CODE_REQUEST.search(query) else '')+''.join('\n\n'+x for x in transport_additions(query))


_MUTABLE_DEFAULT = re.compile(r"(?is)def\s+(?P<fn>[a-zA-Z_]\w*)\s*\(\s*(?P<x>[a-zA-Z_]\w*)\s*,\s*(?P<lst>[a-zA-Z_]\w*)\s*=\s*\[\s*\]\s*\)\s*:\s*(?P=lst)\.append\(\s*(?P=x)\s*\)\s*;?\s*return\s+(?P=lst)\b")

def mutable_default_solution(query: str):
    """Narrow append-and-return pattern; preserve the original function API."""
    import keyword
    m=_MUTABLE_DEFAULT.search(query or '')
    if not m or not re.search(r'(?i)اشرح|المشكلة|راجع|صلح|correct|fix|bug|pytest|اختبار',query):return None
    fn,x,lst=m.group('fn','x','lst')
    if len({x,lst})!=2 or any(keyword.iskeyword(v) for v in (fn,x,lst)):return None
    negative=bool(re.search(r'(?i)سالب|سالبة|negative',query))
    first=f'def test_negative_value():\n    assert {fn}(-5) == [-5]' if negative else f'def test_first_call():\n    assert {fn}(1) == [1]'
    return f"""**المشكلة:** `{lst}=[]` قائمة افتراضية واحدة مشتركة بين الاستدعاءات التي لا تمرر قائمة، فتتراكم العناصر.

**الإصلاح:** استخدم `None` وأنشئ قائمة جديدة عند الحاجة.
```python
def {fn}({x}, {lst}=None):
    if {lst} is None:
        {lst} = []
    {lst}.append({x})
    return {lst}
```
**3 اختبارات pytest:**
```python
{first}

def test_independent_calls():
    assert {fn}(1) == [1]
    assert {fn}(2) == [2]

def test_supplied_list_is_updated():
    existing = [10, 20]
    result = {fn}(30, existing)
    assert result == [10, 20, 30]
    assert result is existing
```
الاختبارات جاهزة للتشغيل بـ`pytest`؛ لم أدّعِ تشغيلها في جلسة الشات."""


_AVERAGE_BUG = re.compile(r'(?is)def\s+average\s*\(\s*numbers\s*\)\s*:\s*return\s+sum\s*\(\s*numbers\s*\)\s*/\s*len\s*\(\s*numbers\s*\)')

def average_empty_solution(query: str):
    """Verified, narrow solution for the exact empty-mean Python task."""
    if not _AVERAGE_BUG.search(query or ''):
        return None
    if not re.search(r'(?i)قائمة\s+فاضي|القائمة\s+تكون\s+فاضية|empty|pytest|قائمة\s+فارغة',query):
        return None
    return '''**المشكلة:** لو `numbers=[]` فإن `len(numbers)=0`، والدالة الأصلية هترفع `ZeroDivisionError`. هنختار سلوك واضح: `ValueError` عند القائمة الفارغة.

```python
def average(numbers):
    if not numbers:
        raise ValueError("Cannot calculate average of an empty list")
    return sum(numbers) / len(numbers)

# pytest tests
import pytest

def test_average_normal():
    assert average([1, 2, 3]) == 2.0

def test_average_empty():
    with pytest.raises(ValueError, match="empty"):
        average([])

def test_average_negative():
    assert average([-1, -2, 3]) == 0.0
```
الاختبارات دي اتكتبت عشان تُشغّل في `pytest`، ومش هادّعي إنها اتنفذت عند المستخدم.'''


def exact_short_calculation(query: str):
    """Only explicit simple integer calculation with output-only constraint."""
    import operator
    q=(query or '').strip().translate(str.maketrans('٠١٢٣٤٥٦٧٨٩','0123456789'))
    m=re.fullmatch(r'(?i)(?:احسب|calculate)\s+(-?\d{1,8})\s*([×*+−-])\s*(-?\d{1,8})\s*(?:[،,]\s*)?(?:واكتب\s+(?:الرقم|الناتج)\s+فقط|(?:return|write)\s+(?:the\s+)?(?:number|result)\s+only)[.!؟»"]*',q)
    if not m:return None
    a,op,b=m.groups()
    return str({'×':operator.mul,'*':operator.mul,'+':operator.add,'−':operator.sub,'-':operator.sub}[op](int(a),int(b)))


def stable_unique_solution(query: str):
    """Known safe repair only for list(set(items)) with a first-order requirement."""
    if not re.search(r'(?i)ترتيب\s+(?:أول|اول)\s+ظهور|first[- ](?:appearance|occurrence)|preserv\w*\s+(?:the\s+)?order',query or ''):return None
    m=re.search(r'(?s)def\s+([A-Za-z_]\w*)\((\w+)\)\s*:\s*return\s+list\(set\(\2\)\)',query)
    if not m:return None
    import keyword
    fn,x=m.groups()
    if not x.isidentifier() or any(keyword.iskeyword(v) for v in (fn,x)):return None
    return f"""`set` لا يضمن ترتيب أول ظهور. للعناصر القابلة للـhash استخدم `dict.fromkeys`، أو حلقة بمقارنة القيم لو العناصر غير قابلة للـhash. الكود التالي يحافظ على الترتيب للعناصر القابلة للـhash.
```python
def {fn}({x}):
    return list(dict.fromkeys({x}))
```
```python
def test_duplicates():
    assert {fn}([3, 1, 3, 2]) == [3, 1, 2]

def test_empty():
    assert {fn}([]) == []

def test_no_duplicates():
    assert {fn}([3, 2, 1]) == [3, 2, 1]
```
لم أشغّل هذه الاختبارات داخل جلسة الشات."""
