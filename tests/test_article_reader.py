import unittest
from src.article_reader import public_http_url, publisher_url, readable_article, enrich_articles
from src.realtime import WebEvidence, NewsPlan, evidence_prompt, news_headline_digest, evidence_footer
from datetime import datetime, timezone

now = datetime(2026,11,19,tzinfo=timezone.utc)
plan=NewsPlan('أخبار امبارح','political news',['news'],now,now,True)

class ArticleReaderTests(unittest.TestCase):
    def test_redirect_to_private_url_is_not_followed(self):
        from unittest.mock import patch
        class Resp:
            status_code=302
            headers={'Location':'http://127.0.0.1/admin'}
            def close(self):pass
        with patch('requests.get',return_value=Resp()) as get:
            self.assertEqual(readable_article('https://apnews.com/article/test'),'')
        self.assertEqual(get.call_count,1)
        self.assertFalse(get.call_args.kwargs['allow_redirects'])

    def test_public_url_guard(self):
        for u in ['http://localhost/','http://127.0.0.1/','file:///tmp/secret','http://169.254.169.254/latest','javascript:alert(1)']:
            self.assertFalse(public_http_url(u),u)
        self.assertTrue(public_http_url('https://apnews.com/article/example'))
    def test_google_news_resolution_to_publisher(self):
        u='https://news.google.com/rss/articles/opaque123'
        self.assertEqual(publisher_url(u,decoder=lambda url,**kwargs:'https://apnews.com/article/legitimate'),
                         'https://apnews.com/article/legitimate')
        self.assertEqual(publisher_url(u,decoder=lambda url,**kwargs:'http://127.0.0.1/internal'),'')
    def test_readable_text_minimum(self):
        content='Test article paragraph about trade and economic negotiations. '*10
        result=readable_article('https://apnews.com/article/example',fetch=lambda u:('<html></html>',u),
                                extractor=lambda html,url:content)
        self.assertIn('trade',result)
        self.assertEqual(readable_article('https://apnews.com/article/example',fetch=lambda u:'bad',
                                           extractor=lambda html,url:'short'),'')
    def test_enrichment_is_bounded_and_marks_status(self):
        src=[]
        for i in range(4):
            src.append({'title':f'Story {i}', 'publisher':'AP', 'published':'2026-11-18 18:00',
                        'url':f'https://news.google.com/rss/articles/id{i}', 'snippet':''})
        ev=WebEvidence('found','2026-11-19 02:00','Google News RSS',src,plan=plan)
        body='Trade and politics article. '*30
        enrich_articles(ev,max_articles=2,decoder=lambda u,**kw:u.replace('news.google.com/rss/articles','apnews.com/article'),
                        fetch=lambda u:('<html><article></article></html>',u),
                        extractor=lambda html,url:body)
        self.assertEqual([s['read_status'] for s in ev.sources],
                         ['article_read','article_read','headline_only','headline_only'])
        self.assertIn('ARTICLE_BODY_READ',evidence_prompt(ev))
        self.assertIn('HEADLINE_ONLY',evidence_prompt(ev))
        self.assertIn('مقتطف قصير',news_headline_digest(ev))
        self.assertIn('محتوى المقال متاح',evidence_footer(ev))
    def test_failed_access_never_fabricates_body(self):
        src=[{'title':'Fake title','url':'https://news.google.com/rss/articles/abc',
              'published':'2026-11-18 18:00','publisher':'Example','snippet':''}]
        ev=WebEvidence('found','now','Google News RSS',src,plan=plan)
        enrich_articles(ev,decoder=lambda *a,**k:None)
        self.assertEqual(src[0]['read_status'],'headline_only')
        self.assertNotIn('article_text',src[0])
        self.assertIn('العنوان فقط',news_headline_digest(ev))

if __name__=='__main__': unittest.main()
