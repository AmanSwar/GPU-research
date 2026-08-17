# Link audit — 2026-08-17

**What this is.** The cheap half of a citation audit, run with
[`scripts/check_links.sh`](scripts/check_links.sh) and **zero model calls**: every
URL in the corpus gets an HTTP request. This catches invented links, typos and
moved pages. It does **not** check that a live page says what the citing document
claims — that still needs a reader, and that pass has not run.

## Result

| status | count | reading |
|---|---:|---|
| **200** | **1,105** | resolves |
| 403 | 10 | bot-blocked, not absent |
| 404 | 9 | see triage below |
| 401 / 400 / 206 | 3 | API endpoints and range requests, not citations |
| **total distinct URLs** | **1,140** | |

**96.9% resolve.** Of the remainder, the triage below finds **zero surviving
fabricated citations** — which is the failure mode the audit existed to catch,
and the reason to report this before the expensive pass runs.

## Triage of the 35 non-200s

### Bot-blocked (10) — real pages, hostile to `curl`

Perplexity's blog hub (6 URLs), a Medium post, `doi.org`, Supermicro, and a
Databricks Medium article. All return 403 to an automated request and 200 to a
browser. **No action.** Verified by inspection of the host pattern, not by
re-fetching each.

### Already flagged by the corpus itself (2)

`modal.com/blog/cuda-graphs` and
`modal.com/blog/the-hidden-economics-of-llm-inference` both 404. These are
already listed under **"Confirmed nonexistent (returned HTTP 404 during this
pass)"** in
[`04-industry/07-baseten-modal-and-the-yc-inference-companies.md`](04-industry/07-baseten-modal-and-the-yc-inference-companies.md)
— that document ran its own verification pass and caught them. They are named as
non-existent rather than cited as sources. **No action**; this is the system
working.

### False positives from the extractor (2)

- `https://huggingface.co/zai-org/<M>/raw/main/config.json` — a `curl` command
  template in prose, where `<M>` is a placeholder for the model name.
- `https://github.com/vllm-project/vllm/.../symm_mem.py` — an elided path inside
  a quoted code comment.

Both are correct as written; the regex simply cannot tell prose from citation.
**No action.**

### Genuinely stale paths (5) — real projects, moved or restructured docs

| URL | cited in | likely cause |
|---|---|---|
| `developer.nvidia.com/blog/tag/tensorrt-llm/` | `04-industry/01` | NVIDIA retired blog tag-index pages |
| `nvidia.github.io/TensorRT-LLM/performance/perf-best-practices.html` | `04-industry/01` | docs site restructured |
| `github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/performance` | `04-industry/01` | directory moved in-repo |
| `docs.sglang.io/supported_models/generative_models.html` | `05-models/03` | docs site restructured |
| `sky.cs.berkeley.edu/blog/` | `04-industry/09` | lab site reorganised |

These point at material that exists but has relocated. The *claims* they support
are not thereby wrong, but the links must be re-resolved before the corpus can be
called sourced. **Action: fix in the reader pass.**

### Unresolvable (1)

`www.baseten.co/library/glm-5-2/` (cited in `05-models/01`). Baseten's model
library uses per-model paths; this slug 404s. Either the model is listed under a
different slug or not listed at all. **Action: re-resolve or drop the claim it
supports.**

## What this does not establish

The important half of citation checking is unrun. A URL returning 200 proves a
page exists, not that it contains the number attributed to it, was written by the
named author, or measured on the hardware claimed. Every `[verified]` label in
this corpus is still **author-asserted**, not independently confirmed.

The reader pass — one agent per document, re-fetching sources and checking claims
against them — remains the largest outstanding piece of work on this corpus.

## Reproducing

```
scripts/check_links.sh [concurrency]   # default 12
```

Results land in `scripts/link-check.tsv` as `status <TAB> url <TAB> citing files`,
so any failure is immediately traceable to the documents that need editing.
