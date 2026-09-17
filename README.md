[English](README.md) | [中文](README.zh-CN.md)

# Chinese-English Mixed-Document RAG QA System

A retrieval-augmented QA system with PDF / DOCX / TXT / MD parsing, **language-aware
embeddings**, FAISS dual-index retrieval, and cross-lingual RRF fusion. It ships with a
pure-retrieval mode that needs no LLM service at all — clone it and you get results.

---

## ⚠️ What This Repo Is (and Isn't)

**This is an open-source reproduction of the "language-aware retrieval" method — not the code of
any original project.**

- The original project ran on a **self-collected dataset** and **edge-device** hardware. That data
  and that hardware are not here, and cannot be recovered.
- This repo rebuilds the same method with **public models + a bundled demo corpus (8 Chinese and
  English documents, 35 chunks)**. The metrics produced here are therefore **not directly
  comparable to any original project's**.
- Every number in this repo was measured on the **corpus bundled with this repo**, and can be
  reproduced with a single command.

This section exists because: a repo that claims to reproduce a set of metrics and then can't is
worse than no repo at all.

---

## Conclusion First

My original hypothesis was: "route Chinese to a Chinese model and English to an English model, and
retrieval will surely beat a single multilingual model." Testing showed it was **only half right** —
and it was wrong in an interesting way.

Corpus: 8 Chinese/English documents, 35 chunks (15 Chinese / 20 English). Evaluation: 20 questions,
12 same-language and 8 cross-lingual (asked in Chinese with the answer only in an English document,
or the reverse).

| Approach | Same-lang R@1 | Cross-lingual R@1 | Cross-lingual R@5 | Overall R@1 | Overall R@5 |
|---|---|---|---|---|---|
| Multilingual single model (control) | 0.833 | **1.000** | **1.000** | **0.900** | 0.950 |
| Language routing | **1.000** | 0.000 | 0.000 | 0.600 | 0.600 |
| Equal-weight RRF | 0.500 | 0.375 | **1.000** | 0.450 | **1.000** |
| Weighted RRF (w=1.5) | **1.000** | 0.000 | 0.000 | 0.600 | 0.600 |

### Three Findings

**1. Language routing was the only approach to score a perfect same-language result — at the cost of
taking cross-lingual all the way to zero.**

Routing each query to the monolingual sub-index for its language pushed same-language R@1 to
1.000 — the only one of the four approaches to manage it. But all 8 cross-lingual questions scored 0:
routing **completely excludes** documents in the other language from the candidate set. It isn't that
they rank badly; they aren't in the candidate set at all.

**2. RRF rescues recall, but not precision.**

Equal-weight RRF pulled cross-lingual R@5 back from 0 to **1.000**, but same-language R@1 fell from
1.000 to 0.500. The reason is that RRF looks only at rank, not at the score distribution: a chunk
ranked "#1 in its own language, #20 in the other" and a chunk ranked "#1 in one stream, entirely
absent from the other" receive almost the same RRF score. Each space returns 20 candidates while
each space holds only 15–20 chunks, which means **each space returned all of its content** — so rank
position stops being a strong signal.

**3. The attempt to fix this with weighting failed — and it failed cleanly.**

| primary_weight | Same-lang R@1 | Cross-lingual R@5 | Overall R@1 | Overall R@5 |
|---|---|---|---|---|
| 1.00 | 0.500 | 1.000 | 0.450 | 1.000 |
| 1.05 | 1.000 | 0.875 | 0.600 | 0.950 |
| 1.10 | 1.000 | 0.000 | 0.600 | 0.600 |
| 1.50 | 1.000 | 0.000 | 0.600 | 0.600 |

This is a **step function**, not a trade-off curve. Either the weight is ≤ 1.0 (which is the same as
not adding one — it degenerates into equal-weight RRF), or, as soon as it is slightly above 1.0, the
primary-language stream's rank-1 permanently outranks the other stream's rank-1 and it degenerates
straight into pure routing. There is no usable middle ground.

### So What's the Conclusion

**On this corpus, the plain multilingual single model is actually the strongest single choice
(overall R@1 0.900).**

The value of language-aware routing isn't in these retrieval metrics — it's elsewhere:

- **Index and latency**: the small monolingual models (24M / 33M) are faster and use less memory
  than the multilingual model (118M);
- **Swappability**: you can drop in a stronger model per language instead of settling for one
  jack-of-all-trades;
- **Purely monolingual settings**: if 100% of your traffic is Chinese queries, routing really is
  better on same-language retrieval.

In other words: **language routing is a component, not a solution.** Whether to use it depends on the
language distribution of your queries, and that has to be measured before it can be decided — which
is exactly what this repo does.

### ⚠️ A Methodological Problem That Has to Be Stated

**The comparison above is not parameter-matched.** The multilingual baseline
`paraphrase-multilingual-MiniLM-L12-v2` has **117.7M** parameters, while `bge-small-zh` +
`bge-small-en` together have only **57.4M**. The baseline gets a free ride on capacity, and this repo
**cannot distinguish** how much of "the multilingual model is stronger" comes from architecture and
how much from parameter count.

To really settle it, the run has to be repeated with a multilingual model of comparable size
(~60M). That is the first thing to do next.

---

## Quick Start

```bash
pip install -r requirements.txt

# 1. Generate the bundled demo corpus (8 Chinese/English documents, covering md/txt/docx/pdf)
python scripts/build_demo_corpus.py

# 2. Ask a question — pure-retrieval mode by default, no LLM service required
python scripts/ask.py "本科生一次最多能借几本书?"

# 3. To generate answers, hook up a local Ollama (free, no API key needed)
ollama pull qwen2.5:7b
python scripts/ask.py "食堂能用现金付款吗?" --llm ollama

# 4. Reproduce every evaluation number above
python eval/evaluate.py
python eval/evaluate.py --modes routed,cross_lingual   # skip the 470MB multilingual baseline

# 5. Web demo
pip install streamlit && streamlit run app.py

# 6. Tests (40 cases, no model downloads needed)
python -m pytest tests -q
```

The first run downloads the embedding models from HuggingFace: `bge-small-zh-v1.5` at about 95MB,
`bge-small-en-v1.5` at about 130MB, and the multilingual baseline at about 470MB. Models load on
demand — if a path isn't used, it isn't downloaded.

### One Illustrative Failure Case

```console
$ python scripts/ask.py "图书馆周末几点开门?" --mode hybrid
问题     图书馆周末几点开门?
查询语种  zh
检索空间  ['zh', 'en']
置信度    0.547

检索结果 top-5  (分数列 = rrf)
  [1]   0.0246  图书馆借阅规则.md  (zh)   ← it snatched the slot
  [2]   0.0242  宿舍管理条例.txt  (zh)
  [3]   0.0238  图书馆借阅规则.md  (zh)
  ...
```

The correct answer is in the English `library_policy.md` ("On weekends the library opens at 10:00
and closes at 18:00"), but because the query is in Chinese, weighted RRF lets the Chinese
图书馆借阅规则 (library borrowing rules) document override the English one across the board. That is
what "weighted RRF cross-lingual R@5 = 0" in the table above looks like in practice.

---

## Directory Structure

```
multilingual-rag-qa/
├── _bootstrap.py            # shared by entry scripts: inject src path + switch to UTF-8 output
├── app.py                   # Streamlit demo UI
├── requirements.txt
├── src/rag/
│   ├── config.py            # every hyperparameter lives here, for reproducibility
│   ├── language.py          # language detection (CJK character ratio)
│   ├── loaders.py           # PDF / DOCX / TXT / MD → paragraph chunks
│   ├── chunking.py          # chunking with a per-language character budget
│   ├── embedders.py         # language-aware embeddings + multilingual baseline
│   ├── index.py             # FAISS dual index + RRF fusion
│   ├── prompts.py           # dynamic prompt engineering
│   ├── llm.py               # NullLLM (pure retrieval) / OllamaLLM
│   └── pipeline.py          # end-to-end flow
├── scripts/
│   ├── build_demo_corpus.py # generate the demo corpus (includes a hand-rolled minimal PDF writer)
│   ├── ingest.py            # build the index and save it to disk
│   └── ask.py               # ask questions from the command line
├── eval/
│   ├── qa_set.jsonl         # 20 labeled questions
│   └── evaluate.py          # recall@k / MRR, broken down by same-language and cross-lingual
├── tests/                   # 40 cases, none of which need model downloads
└── data/raw/                # the 8 demo documents
```

---

## Three Design Points

### 1. Language-aware chunking, not just language-aware embeddings

A Chinese character carries roughly twice the information density of an English character. Using the
same character cap for both would make Chinese chunks carry too much content and English chunks too
fragmented, and the retrieval granularity on the two sides would no longer be comparable. So the
character budget is split by language: **200 Chinese characters / 420 English characters** — only
then do the two land in roughly the same token range. `ChunkConfig.budget()` is the single entry
point for this logic.

### 2. Why RRF instead of comparing scores directly

The Chinese model and the English model produce cosine scores on **different scales**, so comparing
them directly is simply wrong. RRF depends only on rank and not on score, which sidesteps the problem
by construction — that's the reason it exists.

But "Finding 2" above also shows what that costs: the price of throwing away the score distribution
is that you can no longer tell "rank 1 in one stream, rank 20 in the other" apart from "rank 1 in one
stream, absent from the other." **This is inherent to RRF, not an implementation defect.**

### 3. Dynamic prompt engineering — three decidable signals

"Dynamic" is not an adjective here; it means three signals that materially change the text sent to
the model:

| Signal | Effect |
|---|---|
| Query intent (factual / summary / comparison) | Switches to a completely different answer structure and set of examples |
| Retrieval confidence below threshold | Appends an instruction that **explicitly permits answering "cannot determine"** |
| Hits spanning multiple languages | Requires answering in the language of the question and labeling the language of each cited source |

The second one is the most effective hallucination suppressor of the three: without an explicit
escape hatch in the instructions that allows saying "I don't know," the model will almost always
fabricate an answer.

---

## Reproducing the Evaluation

```bash
python eval/evaluate.py --show-misses --json eval/result.json
```

- `--show-misses` prints every missed case one by one
- `--primary-weight` adjusts the primary-language weight for weighted RRF
- `--modes` selects which approaches to compare

**The evaluation measures retrieval only and never calls an LLM** — otherwise generation randomness
would contaminate the metrics. The whole pipeline does no random sampling: the same input
necessarily produces the same numbers.

---

## Known Limitations

1. **Parameter counts are not matched** (most important, see above). A 117.7M multilingual baseline
   vs 57.4M for the two monolingual models combined.
2. **The corpus is too small.** 8 documents, 35 chunks, 20 questions. Read the metrics for trends,
   not for decimal places. Each space holds only 15–20 chunks while the candidate set is 20 — i.e.
   every space returns all of its content, which weakens RRF's rank signal (see "Finding 2").
3. **Language detection is heuristic.** Purely numeric or purely symbolic text gets classified as
   English; genuine scenarios with three or more languages need a different approach.
4. **Cross-lingual retrieval only does "run the same question through both languages."** It does not
   use a multilingual aligned space (such as LaBSE / bge-m3), and it does not do query translation.
5. **There is no rerank stage.** Adding cross-encoder reranking would very likely improve the
   metrics on both sides — and it's a potential remedy for the current 0.500 same-language R@1.
6. **`it_policy.pdf` is a hand-constructed pure-ASCII PDF**, because a Chinese PDF would require
   embedding CID fonts.

---

## Next Steps

In priority order:

1. **Re-run with parameter counts matched** — rule out the "baseline wins on parameter count"
   explanation, without which the conclusions above don't hold.
2. **Score-normalized fusion** — z-score normalize within each space first, then fuse. In theory
   this preserves both same-language precision and cross-lingual recall, and it is a direct
   replacement for RRF.
3. **Add cross-encoder rerank** — rerank the fused candidates and go straight at R@1.
4. **Enlarge the evaluation set** — 20 questions is far too wide a confidence interval.

---

## License

MIT
