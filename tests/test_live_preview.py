from threading import Event
from unittest.mock import patch
from src.dialogue import fresh_memory, build_messages
from src.quality_next import structured_stream
from src.job_manager import _ACTIVE_CANCEL
import app
from tests.test_quality_next import PLAN, GOOD, BAD

def test_first_chunk_visible_before_generation_finishes():
    events=[]
    def gen(*args):
        yield GOOD[:60]
        events.append('finished')
        yield GOOD
    out=structured_stream(PLAN,fresh_memory(),gen,build_messages,500,live_preview=True)
    first=next(out)
    assert GOOD[:60] in first and 'مسودة أولية' in first and not events
    assert list(out)[-1]==GOOD and events==['finished']

def test_retry_progress_and_clean_final():
    calls=[]
    def gen(*args):
        calls.append(1)
        yield BAD if len(calls)==1 else GOOD
    out=list(structured_stream(PLAN,fresh_memory(),gen,build_messages,500,live_preview=True))
    assert len(calls)==2 and any('محاولة تصحيح واحدة' in a for a in out)
    assert out[-1]==GOOD

def test_cancellation_during_preview_no_retry():
    event=Event();calls=[]
    def gen(*args):
        calls.append(1);yield BAD;yield GOOD
    it=structured_stream(PLAN,fresh_memory(),gen,build_messages,500,event,live_preview=True)
    assert 'مسودة أولية' in next(it)
    event.set();assert list(it)==[] and len(calls)==1

def test_ui_path_enables_preview():
    token=_ACTIVE_CANCEL.set(Event())
    try:
        out=list(app.fast_guarded_stream(PLAN,fresh_memory(),lambda *_:iter([GOOD[:50],GOOD])))
        assert 'مسودة أولية' in out[0] and out[-1]==GOOD
    finally:_ACTIVE_CANCEL.reset(token)

def test_turn_only_commits_final():
    token=_ACTIVE_CANCEL.set(Event())
    try:
        with patch('torch.cuda.is_available',return_value=True),patch.object(app,'warm_model_in_background',return_value=None),patch.object(app,'model_stream',return_value=iter([GOOD[:50],GOOD])):
            out=list(app.turn(PLAN,[],fresh_memory()))
        assert any('مسودة أولية' in x[0][-1]['content'] for x in out)
        assert out[-1][0][-1]['content']==GOOD
        turns=[t for topic in out[-1][2]['topics'] for t in topic['turns'] if t['role']=='assistant']
        assert [t['text'] for t in turns]==[GOOD]
    finally:_ACTIVE_CANCEL.reset(token)
