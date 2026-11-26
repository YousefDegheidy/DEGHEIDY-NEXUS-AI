import unittest
from unittest.mock import patch
from datetime import datetime
from zoneinfo import ZoneInfo
from src.realtime import WebEvidence, NewsPlan, search_web
from src.evidence_guard import (
    evaluate_news_answer, render_news_response, safe_digest,
    impact_question, scenario_answer, _event_state_conflict,
)
from src.dialogue import fresh_memory
import app

PLAN=NewsPlan('اخبار امبارح ترامب الصين','Trump visit China',
              ['Trump China visit'],datetime(2026,11,18,tzinfo=ZoneInfo('Africa/Cairo')),
              datetime(2026,11,19,tzinfo=ZoneInfo('Africa/Cairo')),True)

def ev_obj():
    src=[{'title':'Trump arrived in China for summit; Xi to meet Trump on Thursday',
          'article_text':'President Trump arrived in Shenzhen on Wednesday. The president was welcomed by officials. Leaders will meet on Thursday to discuss trade policy. No agreement has yet been announced. Another important sentence about the state visit and new trade relations.',
          'read_status':'article_read','publisher':'Example News','published':'2026-11-18 21:00 EET',
          'url':'https://example.org/trump-china','article_url':'https://example.org/trump-china',
          'date_kind':'published','snippet':''}]
    return WebEvidence('found','2026-11-19 02:00 EET','test',src,plan=PLAN)

class EvidenceGuardTests(unittest.TestCase):
    def test_true_source_cited_claim_is_accepted(self):
        r=evaluate_news_answer('Trump arrived in China on Wednesday [1].',ev_obj())
        self.assertTrue(r.approved,r.problems)
    def test_future_meeting_is_not_relabelled_completed(self):
        r=evaluate_news_answer('Trump met Xi on Wednesday [1].',ev_obj())
        self.assertFalse(r.approved)
    def test_arrived_vs_on_the_way_conflict(self):
        r=evaluate_news_answer('Trump was on his way to China on Wednesday [1].',ev_obj())
        self.assertFalse(r.approved)
    def test_no_signed_agreement_when_not_announced(self):
        r=evaluate_news_answer('Trump signed an agreement on Wednesday [1].',ev_obj())
        self.assertFalse(r.approved)
    def test_invented_weekday_is_blocked(self):
        r=evaluate_news_answer('Trump arrived in China on Monday [1].',ev_obj())
        self.assertFalse(r.approved)

    def test_uncited_claim_is_not_trusted(self):
        r=evaluate_news_answer('Trump arrived in China on Wednesday.',ev_obj())
        self.assertFalse(r.approved)
    def test_arbitrary_secret_talks_not_trusted(self):
        r=evaluate_news_answer('حصلت محادثات سرية عن اتفاق ترامب مع الصين [1].',ev_obj())
        self.assertFalse(r.approved)
    def test_out_of_range_citation_is_rejected(self):
        r=evaluate_news_answer('Trump arrived in China on Wednesday [10].',ev_obj())
        self.assertFalse(r.approved)
    def test_invalid_answer_is_replaced_with_attributable_digest(self):
        out,report=render_news_response('ايه اخبار زيارة ترامب الصين',
            'Trump met Xi and signed a deal [1].',ev_obj())
        self.assertFalse(report.approved)
        self.assertIn('المؤكد',out)
        self.assertNotIn('signed a deal',out)
    def test_valid_answer_uses_footer(self):
        out,r=render_news_response('اخبار زيارة ترامب الصين',
            'Trump arrived in China on Wednesday [1].',ev_obj())
        self.assertTrue(r.approved)
        self.assertIn('https://example.org/trump-china',out)
    def test_impact_conditional_not_claim_of_confirmed_policy(self):
        out=scenario_answer(ev_obj())
        self.assertIn('قناة السويس',out)
        self.assertIn('تحليل مشروط',out)
        self.assertIn('التمويل',out)
        self.assertNotIn('مشروع نيل',out)
    def test_impact_query(self):
        self.assertTrue(impact_question('طب ده هيأثر على مصر إزاي؟'))
        self.assertFalse(impact_question('كيف يؤثر على الهند؟'))
    def test_low_quality_social_index_not_counted(self):
        t=datetime(2026,11,19,2,0,tzinfo=ZoneInfo('Africa/Cairo'))
        src={'title':'Trump talks with Xi in China','url':'https://news.google.com/rss/articles/abc',
             'published':'2026-11-18T16:00:00Z','publisher':'Facebook'}
        x=search_web('اخبار ترامب الصين امبارح',now=t,search_fn=lambda _: [src])
        self.assertNotEqual(x.status,'found')
    def test_photo_gallery_is_not_evidence_for_normal_news(self):
        t=datetime(2026,11,19,2,0,tzinfo=ZoneInfo('Africa/Cairo'))
        src={'title':'Trump trip to China in photos','url':'https://apnews.com/photo-gallery/photos',
             'published':'2026-11-18T16:00:00Z','publisher':'AP News'}
        x=search_web('اخبار ترامب الصين امبارح',now=t,search_fn=lambda _: [src])
        self.assertNotEqual(x.status,'found')

    def test_followup_impact_reuses_news_and_no_new_search(self):
        def fake_torch():return False
        with patch('torch.cuda.is_available',side_effect=fake_torch),\
             patch.object(app,'search_web',side_effect=AssertionError('Unexpected search')):
            ans=list(app.turn('طب ده هيأثر على مصر إزاي',[],fresh_memory(),ev_obj()))
        final=ans[-1]
        self.assertIn('قناة السويس',final[0][-1]['content'])
        self.assertIsNotNone(final[3])
    def test_unsafe_draft_never_displayed_during_news_generation(self):
        ev=ev_obj()
        with patch('torch.cuda.is_available',return_value=True),\
             patch.object(app,'search_web',return_value=ev),\
             patch.object(app,'enrich_articles',side_effect=lambda e,**k:e),\
             patch.object(app,'guarded_stream',return_value=iter(['Trump met Xi and signed secret deal [1].'])):
            frames=list(app.turn('اخبار زيارة ترامب للصين امبارح',[],fresh_memory()))
        all_text='\n'.join(f[0][-1]['content'] for f in frames)
        self.assertNotIn('Trump met Xi and signed secret deal',all_text)
        self.assertIn('المؤكد',frames[-1][0][-1]['content'])
    def test_no_evidence_impact_is_honest(self):
        with patch('torch.cuda.is_available',return_value=True):
            frames=list(app.turn('ده هيأثر على مصر ازاي',[],fresh_memory(),None))
        self.assertIn('اذكر اسم الحدث',frames[-1][0][-1]['content'])

if __name__=='__main__':unittest.main()
