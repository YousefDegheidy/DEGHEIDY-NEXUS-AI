import unittest
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from email.utils import format_datetime
from unittest.mock import patch
from src import realtime as rt
from src.dialogue import fresh_memory, commit_answer

NOW=datetime(2026,11,19,1,30,tzinfo=ZoneInfo('Africa/Cairo'))
OLD={'title':'A general political article', 'url':'https://example.com/old-news',
     'published':'2026-08-28T11:00:00Z','publisher':'Source'}
TARGET={'title':'Trump meets Xi in China: latest diplomatic talks',
        'url':'https://apnews.com/article/fake-example-news-item',
        'published':'2026-11-18T15:00:00Z','publisher':'AP'}

class TestNewsRetrieval(unittest.TestCase):
    def test_yesterday_is_previous_calendar_day_in_cairo(self):
        p=rt.plan_search('ايه الاخبار السياسيه اللي حصلت امبارح',now=NOW)
        self.assertEqual(p.start.strftime('%Y-%m-%d %H:%M'),'2026-11-18 00:00')
        self.assertEqual(p.end.strftime('%Y-%m-%d %H:%M'),'2026-11-19 00:00')
        self.assertTrue(p.explicit_day)
        self.assertEqual(p.topic,'political news')
    def test_today_is_19_nov(self):
        p=rt.plan_search('اخبار السياسة النهارده',now=NOW)
        self.assertEqual(p.start.date().isoformat(),'2026-11-19')
        self.assertTrue(p.explicit_day)
    def test_explicit_arabic_date(self):
        self.assertEqual(rt.plan_search('أخبار ١٨ نوفمبر ٢٠٢٦',now=NOW).start.date().isoformat(),'2026-11-18')
    def test_user_yesterday_then_followup_trump_china(self):
        m=commit_answer(fresh_memory(),'ايه الاخبار السياسيه اللي حصلت امبارح','نتائج سياسية')
        q='ط و اخبار عن تررامب و زيييارته للصين'
        self.assertTrue(rt.should_search(q,m))
        p=rt.plan_search(q,memory=m,now=NOW)
        self.assertEqual(p.start.date().isoformat(),'2026-11-18')
        self.assertEqual(p.topic,'Trump visit China')
        self.assertTrue(p.inherited_day)
        self.assertEqual(p.queries[0],'Donald Trump China visit')
    def test_not_search_for_old_personal_year(self):
        self.assertFalse(rt.needs_fresh_search('I studied Python in 2026'))
        self.assertFalse(rt.needs_fresh_search('how to become a senior AI engineer'))
    def test_not_accept_undated_or_old_portal(self):
        results=[OLD,
                 {'title':'Politics in 2026','url':'https://en.wikipedia.org/wiki/2026_in_politics','published':'2026-11-18'},
                 {'title':'أخبار السياسة اليوم','url':'https://asharq.com/politics/','published':None}]
        ev=rt.search_web('أخبار إمبارح',now=NOW,search_fn=lambda _:results)
        self.assertEqual(ev.status,'no_results')
        self.assertFalse(ev.sources)
        self.assertIn('لا يعني عدم وقوع أحداث',rt.unavailable_message(ev))
    def test_only_matching_date_is_retained(self):
        good={'title':'New parliamentary debate', 'url':'https://example.org/politics-20261118',
              'published':'2026-11-18T18:00:00Z'}
        ev=rt.search_web('ايه الأخبار السياسية امبارح',now=NOW,search_fn=lambda _: [OLD,good])
        self.assertEqual(ev.status,'found')
        self.assertEqual(len(ev.sources),1)
        self.assertIn('2026-11-18',ev.sources[0]['published'])
    def test_followup_filters_unrelated_stories(self):
        m=commit_answer(fresh_memory(),'ايه الاخبار السياسيه اللي حصلت امبارح','أخبار')
        ev=rt.search_web('ط و اخبار عن تررامب و زيييارته للصين',now=NOW,memory=m,
            search_fn=lambda _: [TARGET,{'title':'China basketball tournament','url':'https://example.com/sports', 'published':'2026-11-18'}])
        self.assertEqual(len(ev.sources),1)
        self.assertIn('Trump',ev.sources[0]['title'])
    def test_trump_xi_headline_is_relevant_to_china_visit(self):
        p=rt.plan_search('اخبار زيارة ترامب للصين امبارح',now=NOW)
        item={'title':'Trump meets Xi at diplomatic summit',
              'url':'https://example.org/politics-2026', 'published':'2026-11-18T15:00:00Z'}
        self.assertIsNotNone(rt._source_item(item,p,NOW))
    def test_google_rss_explicit_day_range_is_not_two_days(self):
        called=[]
        class Resp:
            content=b'<rss><channel></channel></rss>'
            def raise_for_status(self):pass
        def mock_get(url,**kwargs):
            called.append(url)
            return Resp()
        p=rt.plan_search('اخبار سياسة امبارح',now=NOW)
        rt._google_news(p,get=mock_get)
        self.assertTrue(called)
        self.assertTrue(all('before%3A2026-11-19' in url for url in called))
    def test_google_rss_parses_published_date(self):
        class Resp:
            def __init__(self):
                self.content = b'''<?xml version="1.0"?><rss><channel><item>
<title>Trump holds talks in China - BBC</title><link>https://news.google.com/rss/articles/fake123</link>
<pubDate>Wed, 18 Nov 2026 15:15:00 GMT</pubDate><source url="https://bbc.com">BBC</source>
<description>Diplomatic talks</description></item></channel></rss>'''
            def raise_for_status(self):pass
        p=rt.plan_search('ترامب والصين امبارح',now=NOW)
        raw=rt._google_news(p,get=lambda *a,**kw:Resp())
        self.assertTrue(raw)
        filtered=rt._dedupe_rank(raw,p,NOW,6)
        self.assertEqual(len(filtered),1)
        self.assertEqual(filtered[0]['publisher'],'BBC')
    def test_gdelt_timestamp_is_indexed_not_published(self):
        class Resp:
            def json(self):return {'articles':[{'url':'https://example.com/pol','title':'Diplomacy and politics', 'seendate':'20261118T130000Z'}]}
            def raise_for_status(self):pass
        p=rt.plan_search('أخبار السياسة امبارح',now=NOW)
        raw=rt._gdelt(p,get=lambda *a,**kw:Resp())
        filtered=rt._dedupe_rank(raw,p,NOW,6)
        self.assertEqual(filtered[0]['date_kind'],'observed')
        ev=rt.WebEvidence('found',NOW.isoformat(),'GDELT',filtered,plan=p)
        self.assertIn('رُصد',rt.evidence_footer(ev))
    def test_unavailable_does_not_claim_news_absent(self):
        p=rt.plan_search('أخبار سياسية امبارح',now=NOW)
        ev=rt.WebEvidence('unavailable',NOW.isoformat(),'none',plan=p)
        text=rt.unavailable_message(ev)
        self.assertIn('مش دليل على عدم وجود أخبار',text)
        self.assertNotIn('لا توجد أخبار سياسية',text)
    def test_rss_date_utc_to_cairo_boundary(self):
        date=rt._date_utc('2026-11-18T22:45:00Z',now=NOW)
        self.assertEqual(date.date().isoformat(),'2026-11-19')
        p=rt.plan_search('أخبار إمبارح',now=NOW)
        self.assertIsNone(rt._source_item({'title':'Political news','url':'https://example.com/p', 'published':'2026-11-18T22:45:00Z'},p,NOW))
    def test_invalid_and_duplicated_urls_rejected(self):
        a={'title':'Trump discusses policy with China','url':'https://example.com/a', 'published':'2026-11-18T14:00:00Z'}
        ev=rt.search_web('ترامب الصين امبارح',now=NOW,search_fn=lambda _: [a,a,dict(a,url='javascript:alert(1)')])
        self.assertEqual(len(ev.sources),1)
    def test_prompt_injection_marked_untrusted(self):
        ev=rt.search_web('ترامب الصين امبارح',now=NOW,search_fn=lambda _:[TARGET])
        self.assertIn('UNTRUSTED',rt.evidence_prompt(ev))
        msgs=[{'role':'system','content':'base'},{'role':'user','content':'question'}]
        out=rt.inject_evidence(msgs,ev)
        self.assertEqual(msgs[0]['content'],'base')
        self.assertIn('Trump',out[0]['content'])

if __name__=='__main__':unittest.main()
