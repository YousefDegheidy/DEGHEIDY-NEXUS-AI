"""Strict leakage and privacy checks on training examples, no external calls."""
from __future__ import annotations

import argparse
import json
import re
import hashlib
from collections import Counter
from pathlib import Path
from training.validate_dataset import load_conversations

PRIVATE_PATTERNS = (
    re.compile(r'(?i)\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,})\b'),
    re.compile(r'(?i)-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(r'(?i)(?:api[_-]?key|access[_-]?token|password)\s*[:=]\s*[\"\']?[A-Za-z0-9+/=_-]{16,}'),
)


def normalized(s: str) -> str:
    return ' '.join(s.casefold().split())


def audit_rows(rows, holdout_questions=()):
    problems=[]
    train_prompts=set()
    role_counter=Counter()
    for i,messages in enumerate(rows,1):
        for m in messages:
            role_counter[m['role']]+=1
            for pattern in PRIVATE_PATTERNS:
                if pattern.search(m['content']):
                    problems.append(f'line {i}: possible private credential in {m["role"]}')
        prompt=normalized(' '.join(m['content'] for m in messages if m['role']=='user'))
        if not prompt: problems.append(f'line {i}: empty prompt')
        if prompt in train_prompts: problems.append(f'line {i}: duplicate user prompts')
        train_prompts.add(prompt)
    for q in holdout_questions:
        if normalized(q) in train_prompts:
            problems.append('holdout question exactly matches training prompt')
    return problems, role_counter


def audit_file(train_path, holdout_path='evaluation/holdout_10.json'):
    rows=load_conversations(train_path)
    holdout=json.loads(Path(holdout_path).read_text(encoding='utf-8')) if Path(holdout_path).exists() else []
    problems, counts=audit_rows(rows,[r['question'] for r in holdout])
    return {'training_rows':len(rows),'role_counts':dict(counts),
            'data_sha256':hashlib.sha256(Path(train_path).read_bytes()).hexdigest(),
            'exact_holdout_overlap':any('holdout' in p for p in problems),
            'issues':problems,'approved_for_training':not problems}


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('--holdout',default='evaluation/holdout_10.json')
    a=p.parse_args(argv)
    report=audit_file(a.dataset,a.holdout)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report['issues']:raise SystemExit(2)


if __name__=='__main__':main()
