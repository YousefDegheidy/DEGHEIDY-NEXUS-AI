"""Syntax and leakage-check foundation for assistant-only conversational SFT."""
from __future__ import annotations
import json
from pathlib import Path

def load_conversations(path):
    rows=[]
    seen=set()
    for i,line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(),start=1):
        if not line.strip():continue
        obj=json.loads(line)
        messages=obj.get('messages')
        if not isinstance(messages,list) or len(messages)<2:
            raise ValueError(f'Line {i}: expected a list of >=2 messages')
        if messages[-1].get('role')!='assistant' or not str(messages[-1].get('content','')).strip():
            raise ValueError(f'Line {i}: last message must be a nonempty assistant answer')
        if not any(m.get('role')=='user' for m in messages):
            raise ValueError(f'Line {i}: no user message')
        for msg in messages:
            if msg.get('role') not in ('system','user','assistant') or not isinstance(msg.get('content'),str):
                raise ValueError(f'Line {i}: invalid role/content')
        question=' '.join(m['content'] for m in messages if m['role']=='user').strip().casefold()
        if question in seen:raise ValueError(f'Line {i}: duplicate user prompt')
        seen.add(question)
        rows.append(messages)
    if not rows:raise ValueError('Empty training data')
    return rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('path')
    a=p.parse_args()
    rows=load_conversations(a.path)
    print(f'Validated {len(rows)} unique conversation examples; quality not assessed.')
