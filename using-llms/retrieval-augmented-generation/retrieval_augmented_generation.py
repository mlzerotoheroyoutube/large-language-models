"""Retrieval-Augmented Generation: code from the ML Zero to Hero explainer.

Video: https://youtu.be/iex3vkcrxIU
"""

# Minimal RAG: sentence-transformers + NumPy retrieval, then a grounded prompt.
# pip install sentence-transformers numpy anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

docs = {
    "hr-4.2": "Full-time employees receive 15 paid sick days per calendar year.",
    "hr-4.3": "Unused vacation days roll over up to a maximum of 5 days.",
    "it-1.1": "Laptops must be encrypted and screens locked after 5 minutes.",
    "hr-7.0": "Parental leave is 16 weeks fully paid for all parents.",
}
ids, texts = list(docs), list(docs.values())

emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
D = emb.encode(texts, normalize_embeddings=True)          # (n_docs, 384)

def retrieve(query, k=2):
    q = emb.encode([query], normalize_embeddings=True)[0]
    scores = D @ q                                         # cosine similarity
    top = np.argsort(-scores)[:k]
    return [(ids[i], texts[i], float(scores[i])) for i in top]

def build_prompt(query, hits):
    sources = "\n".join(f'<source id="{i}">{t}</source>' for i, t, _ in hits)
    return (f"{sources}\n\nAnswer the question using only the sources above. "
            f"Cite source ids in brackets. If the answer is not in the sources, say "
            f"you don't know.\n\nQuestion: {query}")

query = "How many sick days do I get?"
hits = retrieve(query)
for h in hits:
    print(f"{h[0]}  score={h[2]:.3f}")
prompt = build_prompt(query, hits)

# Generation step (any chat LLM works; Anthropic SDK shown)
import anthropic
client = anthropic.Anthropic()
resp = client.messages.create(model="claude-opus-5-5", max_tokens=300,
                              messages=[{"role": "user", "content": prompt}])
print(resp.content[0].text)

# --------------------------------------------------------------------------

# Hybrid retrieval: BM25 + dense, fused with reciprocal rank fusion, then cross-encoder rerank.
# pip install rank-bm25 sentence-transformers
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

bm25 = BM25Okapi([t.lower().split() for t in texts])

def rrf(rankings, k=60):
    fused = {}
    for ranking in rankings:
        for rank, doc_idx in enumerate(ranking, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused, key=fused.get, reverse=True)

query = "laptop encryption policy"
sparse_rank = list(np.argsort(-bm25.get_scores(query.lower().split())))
dense_rank = list(np.argsort(-(D @ emb.encode([query], normalize_embeddings=True)[0])))
candidates = rrf([sparse_rank, dense_rank])[:3]

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
scores = reranker.predict([(query, texts[i]) for i in candidates])
for i, s in sorted(zip(candidates, scores), key=lambda x: -x[1]):
    print(f"{ids[i]}  rerank={s:.2f}  {texts[i]}")
