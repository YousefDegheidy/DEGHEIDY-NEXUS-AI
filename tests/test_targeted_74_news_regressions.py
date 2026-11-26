"""Failures lifted from the 7.4/10 review: news relevance, direct URLs, and provenance.

Only uses dated invented fixture headlines, not a claim about actual events.
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from src.realtime import search_web, plan_search
from src.evidence_guard import compose_coverage_digest

NOW=datetime(2026,11,23,19,30,tzinfo=ZoneInfo('Africa/Cairo'))
PROMPT=('هات أهم خبرين سياسيين دوليين خلال آخر 12 ساعة بتوقيت القاهرة. '
        'لكل خبر تاريخ نشر التقرير، وملخص من سطرين ورابط مباشر للمصدر الأصلي. '
        'استبعد الرأي والتحليلات.')


def record(url,title,hour=18,kind='published',publisher='Example'):
    return {'url':url,'title':title,'published':f'2026-11-23T{hour:02d}:15:00+02:00',
            'publisher':publisher,'date_kind':kind}


def test_international_news_never_selects_local_expats_interview():
    sample=[
      record('https://www.youmesite.example/local',
             'مصر وطن نحمله معنا.. رئيس الاتحاد العالمي للكيانات المصرية بالخارج يكشف حوار خاص',18),
      record('https://publisher.example/world1','European leaders announce new diplomatic talks',18),
      record('https://publisher.example/world2','US and China ministers meet for bilateral summit',19),
      record('https://publisher.example/op','Opinion: Putin changes world order',19),
    ]
    ev=search_web(PROMPT,search_fn=lambda q:sample,now=NOW)
    titles=[s['title'] for s in ev.sources]
    assert len(titles)==2, titles
    assert all('مصر وطن' not in t and 'Opinion' not in t for t in titles)
    assert 'مصر السياسة الحكومة' not in plan_search(PROMPT,now=NOW).queries


def test_international_must_not_claim_two_news_if_only_one_eligible():
    sample=[record('https://local.example/local','رئيس الاتحاد العالمي للكيانات المصرية بالخارج يتحدث عن مصر'),
            record('https://world.example/news','European leaders announce summit')]
    ev=search_web(PROMPT,search_fn=lambda q:sample,now=NOW)
    result=compose_coverage_digest(PROMPT,ev)
    assert len(ev.sources)==1
    assert 'مش هخترع خبرًا ثانيًا' in result


def test_original_article_link_is_not_google_news_aggregator_link():
    sample=[record('https://news.google.com/rss/articles/abc','European leaders announce summit')]
    ev=search_web(PROMPT,search_fn=lambda q:sample,now=NOW)
    out=compose_coverage_digest(PROMPT.replace('ملخص من سطرين','عناوين'),ev)
    assert 'رابط الناشر المباشر غير متاح' in out
    assert '[المقال لدى الناشر](https://news.google.com' not in out


def test_direct_article_link_is_preserved_after_decoding():
    sample=[record('https://news.google.com/rss/articles/abc','European leaders announce summit')]
    ev=search_web(PROMPT,search_fn=lambda q:sample,now=NOW)
    ev.sources[0]['article_url']='https://publisher.example/news/summit'
    out=compose_coverage_digest(PROMPT.replace('ملخص من سطرين','عناوين'),ev)
    assert '[المقال لدى الناشر](https://publisher.example/news/summit)' in out


def test_observed_date_is_not_implicitly_publication_date():
    q=PROMPT.replace('تاريخ نشر التقرير','تاريخ النشر')
    sample=[record('https://world.example/news','European leaders announce summit',kind='observed')]
    ev=search_web(q,search_fn=lambda _:sample,now=NOW)
    assert ev.status=='no_results'


def test_12hour_window_exact_and_old_story_discarded():
    sample=[record('https://old.example/news','US China diplomats met yesterday',hour=6),
            record('https://new.example/news','Russia Ukraine negotiations renewed',hour=19)]
    ev=search_web(PROMPT,search_fn=lambda _:sample,now=NOW)
    assert len(ev.sources)==1
    assert ev.sources[0]['url']=='https://new.example/news'
    assert (NOW-plan_search(PROMPT,now=NOW).start).total_seconds()==43200


def test_world_news_does_not_accept_domestic_budget_story():
    sample=[record('https://local.example/budget','وزير المالية: مشروع الموازنة أولى أهمية للحماية الاجتماعية')]
    ev=search_web(PROMPT,search_fn=lambda _:sample,now=NOW)
    assert not ev.sources
