"""Resumable, short-poll UI job store. No long-running Gradio SSE responses.

Generations run in daemon threads in this Colab process. The browser only polls a
small snapshot. An interrupted share tunnel no longer discards completed output,
provided the Colab Python runtime stays alive. Tokens are unguessable recovery keys.
"""
from __future__ import annotations

import copy
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_JOBS = 12
JOB_TTL_SECONDS = 4 * 3600

@dataclass
class ResponseJob:
    key: str
    chat: list
    memory: dict
    news_state: Any
    phase: str = 'running'
    error: str = ''
    created_at: float = field(default_factory=time.monotonic)
    updated_at: float = field(default_factory=time.monotonic)
    done_at: float | None = None
    cancellation: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def snapshot(self):
        with self.lock:
            return (copy.deepcopy(self.chat), copy.deepcopy(self.memory),
                    copy.deepcopy(self.news_state), self.phase, self.error)

class ResponseJobManager:
    def __init__(self, *, clock=time.monotonic, runner:Callable | None=None):
        self._clock = clock
        self._runner = runner
        self._jobs: dict[str,ResponseJob] = {}
        self._lock = threading.Lock()

    def _cleanup_locked(self):
        now=self._clock()
        for key,job in list(self._jobs.items()):
            if job.done_at is not None and now-job.done_at > JOB_TTL_SECONDS:
                del self._jobs[key]
        completed=[(key,job) for key,job in self._jobs.items() if job.done_at is not None]
        completed.sort(key=lambda pair: pair[1].done_at or 0)
        while len(self._jobs) >= MAX_JOBS and completed:
            key,_=completed.pop(0)
            self._jobs.pop(key,None)

    def submit(self, question, chat, memory, news_state, prior_key=None):
        question=(question or '').strip()
        if not question:
            return None,'السؤال فاضي'
        if len(question)>12000:
            return None,'السؤال أطول من الحد المسموح؛ اختصره إلى 12000 حرف'
        if prior_key:
            prior=self.get(prior_key)
            if prior and prior.phase == 'running':
                return prior_key,'الرد السابق لسه شغال. استخدم إيقاف أو انتظر ظهوره.'
        with self._lock:
            self._cleanup_locked()
            if len(self._jobs)>=MAX_JOBS:
                return None,'الحد الأقصى للجلسات النشطة اتملأ. جرّب بعد انتهاء أحد الطلبات.'
            key=secrets.token_urlsafe(18)
            initial=copy.deepcopy(chat or [])
            initial.extend([{'role':'user','content':question},
                            {'role':'assistant','content':'⏳ بجهز الإجابة...'}])
            job=ResponseJob(key=key,chat=initial,memory=copy.deepcopy(memory or {}),
                            news_state=copy.deepcopy(news_state))
            self._jobs[key]=job
        def run():
            from src.job_manager import _ACTIVE_CANCEL
            token=_ACTIVE_CANCEL.set(job.cancellation)
            try:
                runner=self._runner
                if runner is None:
                    from app import turn
                    runner=turn
                for shown,text,mem,news in runner(question,chat,memory,news_state):
                    if job.cancellation.is_set():
                        break
                    with job.lock:
                        if job.cancellation.is_set():break
                        job.chat=copy.deepcopy(shown)
                        job.memory=copy.deepcopy(mem)
                        job.news_state=copy.deepcopy(news)
                        job.updated_at=self._clock()
                with job.lock:
                    if job.cancellation.is_set():
                        job.phase='cancelled'
                        if job.chat and job.chat[-1].get('role')=='assistant':
                            current=job.chat[-1]['content']
                            job.chat[-1]['content']=(current if not current.startswith(('⏳','✍️','⚙️','🌐','📰','🔎')) else '') + ('' if '⏹️ تم إيقاف الرد.' in current else '\n\n⏹️ تم إيقاف الرد.')
                    else:
                        last = job.chat[-1]['content'] if job.chat else ''
                        job.phase='failed' if str(last).startswith('⚠️') else 'finished'
                        if job.phase=='failed': job.error=str(last)[:180]
                    job.done_at=self._clock()
            except Exception as exc:
                with job.lock:
                    job.phase='failed'
                    job.error=f'{type(exc).__name__}: {str(exc)[:170]}'
                    if job.chat and job.chat[-1].get('role')=='assistant':
                        job.chat[-1]['content']='⚠️ تعذّر إكمال الرد. راجع سجل Colab. '+job.error
                    job.done_at=self._clock()
            finally:
                _ACTIVE_CANCEL.reset(token)
        threading.Thread(target=run,daemon=True,name='degh-reply-'+key[:5]).start()
        return key,'⏳ بجهز الرد...' 

    def get(self,key):
        if not key or not isinstance(key,str):return None
        with self._lock:
            self._cleanup_locked()
            return self._jobs.get(key)

    def stop(self,key):
        job=self.get(key)
        if not job:return False
        with job.lock:
            if job.phase!='running':
                return False
            job.cancellation.set()
            job.phase='cancelled'
            job.updated_at=self._clock()
            job.done_at=self._clock()
            if job.chat and job.chat[-1].get('role')=='assistant':
                current=str(job.chat[-1]['content'])
                if current.startswith(('⏳','✍️','⚙️','🌐','📰','🔎')):
                    current=''
                job.chat[-1]['content']=current+'\n\n⏹️ تم إيقاف الرد.'
        return True


# Per-request context for optional Transformers stopping criteria.
from contextvars import ContextVar
_ACTIVE_CANCEL:ContextVar[threading.Event | None] = ContextVar('degh_cancel_event',default=None)
