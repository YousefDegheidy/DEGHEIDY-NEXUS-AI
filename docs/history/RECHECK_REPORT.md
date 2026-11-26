# QA74 8.10-based rechecked candidate

- Exact baseline: source ZIP embedded in `DEGHEIDY_QA74_QUALITY_NEXT_LIVE_PREVIEW.ipynb` (SHA-256 5b31124a19ad1cd11ce3c6e8c5358e5884a246edae5ba81f2c3076cfb5bc634e). Previous 8.10 is a manual result, not independently replicated.
- Kept model, adapter loader, training data, ASR, streaming, Stop, and core UI. Changes limited to app routing, precision, official/news selection, structural validation, and recall prompt. Added targeted edit repair.
- Compound arithmetic: independently handles cost per square metre, cost reduction and remaining cost. Missing/ambiguous data is reported, not guessed.
- Official checking: product-specific queries and allowlisted NVIDIA/CUDA and Azure domains, filtering irrelevant results. News: conservative analytical-title/metadata detection and two substantive sentences required if requested. Output is still labeled excerpts, not verified paraphrases; event date can remain unknown. No claim of independent event verification.
- RAG: structural checks for incremental indexing, deletion, authorization before retrieval, per-user isolation. Numeric targets checked when explicitly requested. These are heuristic checks, not semantic proofs. Recall prompts exclude assistant turns for global recall.
- Test results: ........................................................................ [ 29%]
................................................................................................................................... [ 83%]
.........................................                                [100%]
244 passed, 13 subtests passed in 7.75s. One of the new tests exercises 1,000 reproducible arithmetic variants. Search and model-stream fixtures are mocked. Not 1,000 real LLM runs. No CUDA, actual LoRA weights, live news retrieval, audio, or Colab public tunnel available for local verification. Actual improvement above 8.10 is unmeasured.
- Full notebook uses a separate project directory and tries to reuse previous verified adapters. Stage 3 checks local and public /config before advertising a public URL. Stage 5 requires two human-reviewed real runs with existing checks and both scores strictly >8.10. Never replace main before approval.
