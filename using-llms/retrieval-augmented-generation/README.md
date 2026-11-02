# Retrieval-Augmented Generation

**Level:** Intermediate  ·  **Topic:** Large Language Models / Using LLMs

▶️ **Watch the video:** https://youtu.be/iex3vkcrxIU

> RAG answers a question by first **retrieving** relevant passages from an external corpus and then **generating** an answer conditioned on them. It gives an LLM fresh, private and citable knowledge without retraining, and most RAG quality problems are retrieval problems.

## The intuition
There are two kinds of exams: closed-book and open-book. In a closed-book exam you rely on memory, and if you forgot a fact, you might guess. In an open-book exam you look up the right page first, then write your answer using what's on it, and you can even say "see page 42".

A plain language model takes every exam closed-book: it only knows what it memorised during training, which may be out of date, and it knows nothing about your private stuff, like your school's rules or your company's documents. **Retrieval-augmented generation** turns it into an open-book exam.

Here's how. Before any questions, chop all your documents into short pieces, say paragraphs, and store them in a searchable index. When someone asks "How many sick days do I get?", the system searches the index, finds the 3 most relevant paragraphs (say, from the HR handbook), pastes them into the prompt, and tells the model: "Answer using only these paragraphs, and say which one you used." The model replies "15 days per year (HR handbook, section 4.2)."

If the handbook changes next week, you update the index, not the model. And if the search finds the wrong paragraphs, the answer will be wrong, so good searching is half the battle.

## Core notes
- Pipeline: **ingest → chunk → embed → index** (offline); **query → retrieve → rerank → assemble prompt → generate → cite** (online).
- **Chunking**: 200–800 tokens with 10–20% overlap is a common start; respect document structure (headings, sections); store metadata (source, date, ACLs).
- **Hybrid retrieval** (BM25 + dense) with reciprocal rank fusion is a robust default; BM25 nails exact terms (IDs, names, error codes).
- **Rerank** the top 50–100 candidates with a cross-encoder and keep the top 3–10 for the prompt: often the single biggest quality win.
- Prompt: delimit sources, number them, instruct "answer only from the sources; cite [n]; say you don't know if absent".
- Evaluate **retrieval** (recall@k, MRR, nDCG; see Ranking Metrics) separately from **generation** (faithfulness, answer correctness, citation accuracy).
- Query transforms: rewriting, multi-query, decomposition, HyDE; useful for vague or multi-hop questions.
- **Freshness and access control** are RAG's killer features: update the index, filter by user permissions at query time.
- Retrieved content is untrusted input: indirect prompt injection is a real attack vector.
- RAG for **knowledge**, fine-tuning for **behaviour**: Choosing Prompting vs RAG vs Fine-Tuning.

## The math
**RAG-Sequence** (Lewis et al., 2020), with top-$k$ retrieved documents $z$:
$$
p(y\mid x) \approx \sum_{z\in \text{top-}k\,p_\eta(\cdot\mid x)} p_\eta(z\mid x) \prod_{t=1}^{|y|} p_\theta(y_t \mid x, z, y_{<t}).
$$
**RAG-Token** marginalises per token: $p(y\mid x)\approx\prod_t \sum_z p_\eta(z\mid x)\,p_\theta(y_t\mid x,z,y_{<t})$. The retriever is $p_\eta(z\mid x)\propto\exp\big(e_d(z)^\top e_q(x)\big)$ with document and query encoders $e_d, e_q$.

**BM25** score of document $D$ for query $Q=\{q_1..q_n\}$:
$$
\mathrm{BM25}(D,Q) = \sum_{i=1}^{n} \mathrm{IDF}(q_i)\, \frac{f(q_i,D)\,(k_1+1)}{f(q_i,D) + k_1\big(1 - b + b\,\tfrac{|D|}{\mathrm{avgdl}}\big)},
$$
where $f(q_i,D)$ is term frequency, $|D|$ document length, $\mathrm{avgdl}$ the mean length, and typically $k_1\in[1.2,2]$, $b=0.75$.

**Reciprocal rank fusion** of rankings $r\in R$ (Cormack et al., 2009):
$$
\mathrm{RRF}(d) = \sum_{r\in R} \frac{1}{k + \mathrm{rank}_r(d)}, \quad k = 60 .
$$

**Recall@k** $= \frac{|\text{relevant} \cap \text{top-}k|}{|\text{relevant}|}$; the generator can only be as good as recall@k allows.

## Run the code
```bash
pip install anthropic numpy rank_bm25 sentence-transformers
python retrieval_augmented_generation.py
```

## Check yourself
<details><summary>Your RAG system gives wrong answers. How do you debug it?</summary>

Separate retrieval from generation. First build a labelled set of queries with their gold passages and measure recall@k: if the right passage isn't retrieved, fix chunking, parsing, embedding model, hybrid search or query rewriting. If it is retrieved but the answer is wrong, inspect the prompt: position of the passage, too many distractors, missing instruction to cite or abstain, or the model overriding context with parametric memory. Add a reranker, reduce k, and evaluate faithfulness (claims supported by context) separately from correctness.
</details>

<details><summary>Why combine BM25 with dense retrieval?</summary>

They fail differently. Dense bi-encoders capture paraphrase and semantic similarity but are weak on rare exact tokens (product codes, error messages, names) and can generalise poorly out of domain, as BEIR showed. BM25 matches exact terms reliably but misses synonyms. Fusing their rankings with reciprocal rank fusion is cheap, needs no score calibration, and typically improves recall over either alone; a cross-encoder reranker then fixes ordering.
</details>

<details><summary>Bi-encoder vs cross-encoder: what's the difference and where does each go?</summary>

A bi-encoder embeds query and document independently, so document vectors can be precomputed and searched with ANN in milliseconds over millions of items, but query and document never interact token-to-token. A cross-encoder feeds the concatenated pair through a transformer and outputs a relevance score, which is much more accurate but costs a forward pass per pair. So bi-encoders (and BM25) do first-stage recall over the corpus, and cross-encoders rerank the top 20–100 candidates.
</details>

<details><summary>How do you choose chunk size?</summary>

Small chunks give precise matches and less noise but can split the answer and lose context; large chunks keep context but dilute the embedding and waste prompt tokens. Start around 200–800 tokens with modest overlap, chunk on structural boundaries, prepend titles or section headers, and tune on a retrieval eval set. Small-to-big strategies retrieve small chunks but feed their parent section to the LLM, getting both precision and context.
</details>

<details><summary>With million-token context windows, is RAG obsolete?</summary>

No. Retrieval is far cheaper per query than processing a whole corpus, keeps latency low, handles corpora larger than any window, supports freshness and per-user access control, and avoids the accuracy loss models show over very long, distractor-heavy contexts. Long context does change the design: you can retrieve larger units (entire documents) and rely less on aggressive chunking. For one-off analysis of a single large document, long context alone can be the simpler choice.
</details>

---
Part of [ML Zero to Hero](https://www.youtube.com/@mlzerotoherousa): every concept explained from intuition to math.