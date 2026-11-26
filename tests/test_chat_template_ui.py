import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app import normalize_model_messages, fit_chat_context, make_ui

class FakeEncode:
    class Enc(dict):
        pass
    def __init__(self): self.calls=[]
    def __call__(self, messages):
        self.calls.append(messages)
        # Replicate the Qwen3.5 template constraint that caused the incident.
        if sum(m['role']=='system' for m in messages)>1 or messages[0]['role']!='system':
            raise ValueError('System message must be at the beginning.')
        class Ids:
            def __init__(self,n):self.shape=(1,n)
        return {'input_ids':Ids(max(1,sum(len(m['content']) for m in messages)//3))}

class ChatTemplateSafetyTests(unittest.TestCase):
    def test_short_conversation_over_budget_does_not_duplicate_system(self):
        msgs=[{'role':'system','content':'Instructions. '*1150},
              {'role':'user','content':'اخبار ترامب في الصين'}]
        fake=FakeEncode()
        result,encoded=fit_chat_context(msgs,fake,4200)
        self.assertEqual([m['role'] for m in result],['system','user'])
        self.assertLessEqual(encoded['input_ids'].shape[-1],4200)
        self.assertGreater(len(fake.calls),1)

    def test_multiple_system_messages_are_merged_first(self):
        raw=[{'role':'user','content':'first'}, {'role':'system','content':'A'},
             {'role':'assistant','content':'reply'},{'role':'system','content':'B'},
             {'role':'user','content':'followup'}]
        cleaned=normalize_model_messages(raw)
        self.assertEqual([m['role'] for m in cleaned],['system','user','assistant','user'])
        self.assertIn('A',cleaned[0]['content'])
        self.assertIn('B',cleaned[0]['content'])
        self.assertEqual(cleaned[-1]['content'],'followup')

    def test_retry_like_conversation_and_long_news_context(self):
        raw=[{'role':'system','content':'System rules. '*300},
             {'role':'user','content':'What happened yesterday?'},
             {'role':'assistant','content':'Draft'},
             {'role':'user','content':'Correct this answer'},
             {'role':'system','content':'Search evidence. '*550}]
        actual,encoded=fit_chat_context(raw,FakeEncode(),4200)
        self.assertEqual(actual[0]['role'],'system')
        self.assertEqual(sum(m['role']=='system' for m in actual),1)
        self.assertEqual(actual[-1]['content'],'Correct this answer')
        self.assertLessEqual(encoded['input_ids'].shape[-1],4200)

    def test_ui_has_no_two_subtitle_markdowns(self):
        ui=make_ui()
        props=[c.get('props',{}) for c in ui.config['components'] if c.get('type')=='markdown']
        values=[str(p.get('value','')) for p in props]
        self.assertTrue(any('DEGHEIDY — Multilingual AI Chatbot' in value for value in values))
        self.assertFalse(any('بحث أخبار مؤرخ' in value or 'الواجهة هتفتح من غير تحميل' in value for value in values))
        self.assertIn('Ask ??', [c.get('props',{}).get('label') for c in ui.config['components'] if c.get('type')=='textbox'])

if __name__=='__main__': unittest.main()
