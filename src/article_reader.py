"""Bounded publisher-article retrieval for a dated news digest.

A Google News RSS entry is not article content. Resolve its publisher URL, then
extract accessible HTML using Trafilatura. Unsupported/paywalled pages remain
explicitly 'headline_only' — never invent a body or fabricate article reading.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from urllib.parse import urlparse, urljoin
import ipaddress
import re

MAX_ARTICLE_CHARS=2000


def public_http_url(value: str) -> bool:
    try:
        p=urlparse(value or '')
        if p.scheme not in ('http','https') or not p.hostname:return False
        host=p.hostname.lower()
        if host in ('localhost','localhost.localdomain') or host.endswith(('.local','.internal','.localhost')):return False
        if host == 'metadata.google.internal':return False
        try:
            ip=ipaddress.ip_address(host)
        except ValueError:
            # No user-provided URL resolution, and no acceptance of single-label hosts.
            return '.' in host
        return ip.is_global
    except (ValueError,TypeError):return False


def publisher_url(link: str, *, decoder=None) -> str:
    if not public_http_url(link):return ''
    host=(urlparse(link).hostname or '').lower()
    if not host.endswith('news.google.com'):return link
    if decoder is None:
        try:
            from gnews_decoder import decode
            decoder=decode
        except Exception:return ''
    try:
        value=decoder(link,timeout=6.0)
        if isinstance(value,dict):value=value.get('decoded_url') or value.get('url')
        if not isinstance(value,str):return ''
        resolved=value.strip()
        if public_http_url(resolved) and 'news.google.com' not in (urlparse(resolved).hostname or ''):
            return resolved
    except Exception:pass
    return ''


def readable_article(url: str, *, fetch=None, extractor=None):
    if not public_http_url(url):return ''
    try:
        if fetch is None:
            import requests
            # Validate EACH redirect BEFORE following it. Do not let article links
            # send a public Colab instance to private addresses through redirects.
            # DNS rebinding is a separate limitation; do not treat this as a full
            # hardened fetch proxy.
            current=url
            html=''
            for _ in range(4):
                if not public_http_url(current):return ''
                r=requests.get(current,headers={'User-Agent':'Mozilla/5.0 (compatible; DEGHEIDYNews/1.0)'},
                               timeout=(3,7),allow_redirects=False,stream=True)
                try:
                    if r.status_code in (301,302,303,307,308):
                        location=r.headers.get('Location','')
                        if not location:return ''
                        current=urljoin(current,location)
                        continue
                    r.raise_for_status()
                    if not public_http_url(r.url):return ''
                    if 'html' not in r.headers.get('Content-Type','text/html').lower():return ''
                    pieces=[];used=0
                    for c in r.iter_content(chunk_size=16384):
                        used+=len(c)
                        if used>1_200_000:break
                        pieces.append(c)
                    html=b''.join(pieces).decode(r.encoding or 'utf-8',errors='replace')
                    break
                finally:
                    r.close()
            else:return ''
        else:
            raw=fetch(url)
            if isinstance(raw,tuple):html,final=raw
            else:html,final=raw,url
            if not public_http_url(final):return ''
        if not html:return ''
        if extractor is None:
            import trafilatura
            extractor=lambda data,source:trafilatura.extract(data,url=source,
                include_comments=False,include_tables=False,favor_precision=True)
        text=extractor(html,url)
        if not text:return ''
        text=re.sub(r'\s+',' ',str(text)).strip()
        # Very short texts are often banners/landing pages, not articles.
        return text[:MAX_ARTICLE_CHARS] if len(text)>220 else ''
    except Exception:
        return ''


def enrich_articles(evidence, *, max_articles=3, decoder=None, fetch=None, extractor=None):
    """Mutate evidence SOURCES with article content/status; no blind citation claims.

    Decode/fetch at most the top N articles concurrently. Offline/unit tests may
    inject decoder, fetch and extractor; live use never sends text to external LLMs.
    """
    if evidence is None or evidence.status!='found':return evidence
    sources=evidence.sources
    for s in sources:
        s.setdefault('read_status','headline_only')
    max_articles=min(max_articles,len(sources))
    if not max_articles:return evidence
    def one(i,s):
        original=publisher_url(s.get('url',''),decoder=decoder)
        if not original:return i,None,None
        text=readable_article(original,fetch=fetch,extractor=extractor)
        return i,original,text
    with ThreadPoolExecutor(max_workers=min(3,max_articles)) as ex:
        futs=[ex.submit(one,i,sources[i]) for i in range(max_articles)]
        for future in as_completed(futs):
            try:i,original,text=future.result()
            except Exception:continue
            if original:sources[i]['article_url']=original
            if text:
                sources[i]['article_text']=text
                sources[i]['read_status']='article_read'
    return evidence
