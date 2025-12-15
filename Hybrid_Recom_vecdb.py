import os
import re
import gzip
import time
import random
import shutil
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "amazon-meta.txt.gz"   # can also be "amazon-meta.txt"
PERSIST_DIR = "chroma_store"
COLLECTION_NAME = "amazon_products_100k"

RANDOM_SEED = 42
SAMPLE_SIZE = 100_000

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"  # 384-d
EMBED_BATCH = 64                  # CPU batch size for embedding
UPSERT_BATCH = 2000               # avoid chroma max batch errors

# Candidate sizes (tune if needed)
CONTENT_TOP_N = 60                # used in interactive recommend + hybrid
CF_TOP_N = 60

ALPHA = 0.6                       # hybrid knob (0..1)

# Evaluation
KS = (10, 20, 30, 40, 50)
NUM_QUERIES = 50                  # query batch size for eval

# IMPORTANT: always rebuild so embeddings are re-created every run
ALWAYS_FRESH_RUN = True           # deletes persist dir + recreates collection each run


# ============================================================
# FILE OPEN (TXT OR GZ)
# ============================================================

def open_text_file(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="latin-1", errors="ignore")
    return open(path, "rt", encoding="latin-1", errors="ignore")


# ============================================================
# PARSER (amazon-meta format)
# ============================================================

def parse_amazon_meta(path: str):
    """
    Stream-parse amazon-meta file.
    Yields records: {asin,title,group,categories,similar}
    """
    asin = None
    title = ""
    group = ""
    categories = []
    similar = []

    def emit():
        nonlocal asin, title, group, categories, similar
        if asin and title:
            return {
                "asin": asin.strip(),
                "title": title.strip(),
                "group": group.strip(),
                "categories": categories[:],
                "similar": similar[:],
            }
        return None

    with open_text_file(path) as f:
        it = iter(f)
        for raw in it:
            line = raw.strip()

            # New product boundary
            if line.startswith("Id:"):
                rec = emit()
                if rec:
                    yield rec
                asin = None
                title = ""
                group = ""
                categories = []
                similar = []
                continue

            if line.startswith("ASIN:"):
                asin = line.split("ASIN:", 1)[1].strip()
                continue

            if line.startswith("title:"):
                title = line.split("title:", 1)[1].strip()
                continue

            if line.startswith("group:"):
                group = line.split("group:", 1)[1].strip()
                continue

            if line.startswith("categories:"):
                parts = line.split()
                cat_count = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                for _ in range(cat_count):
                    try:
                        cat_line = next(it).strip()
                    except StopIteration:
                        break
                    categories.append(cat_line)
                continue

            if line.startswith("similar:"):
                parts = line.split()
                # format: similar: <n> asin1 asin2 ...
                if len(parts) >= 3:
                    similar = parts[2:]
                continue

        # last record
        rec = emit()
        if rec:
            yield rec


# ============================================================
# RESERVOIR SAMPLING (random 100k)
# ============================================================

def build_random_sample(path: str, sample_size: int, seed: int):
    """
    Reservoir sampling to uniformly sample from a large stream
    without loading everything into memory.
    """
    random.seed(seed)
    sample = []
    n = 0

    for rec in tqdm(parse_amazon_meta(path), desc="Parsing + reservoir sampling"):
        if not rec["asin"] or not rec["title"]:
            continue

        n += 1
        if len(sample) < sample_size:
            sample.append(rec)
        else:
            j = random.randint(1, n)
            if j <= sample_size:
                sample[j - 1] = rec

    return sample


# ============================================================
# TEXT FOR EMBEDDING
# ============================================================

def product_text(rec):
    # Keep concise; include 1 category path if present
    cat = rec["categories"][0] if rec["categories"] else ""
    parts = [rec["title"]]
    if rec["group"]:
        parts.append(f"Group: {rec['group']}")
    if cat:
        parts.append(f"Category: {cat}")
    return " | ".join(parts)


# ============================================================
# CO-PURCHASE GRAPH (within sampled ASINs only)
# ============================================================

def build_copurchase_graph(sample):
    asin_set = set(r["asin"] for r in sample)
    graph = defaultdict(Counter)  # graph[a][b] = weight
    for r in sample:
        a = r["asin"]
        for b in r["similar"]:
            if b in asin_set and b != a:
                graph[a][b] += 1
    return graph


# ============================================================
# CHROMA HELPERS
# ============================================================

def get_chroma_client(persist_dir: str):
    os.makedirs(persist_dir, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
    )


def recreate_collection(client, name: str):
    try:
        client.delete_collection(name)
    except Exception:
        pass
    return client.create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}  # cosine distance
    )


def upsert_in_batches(collection, ids, docs, metadatas, embeddings, batch_size=2000):
    n = len(ids)
    for s in tqdm(range(0, n, batch_size), desc="Upserting into ChromaDB"):
        e = min(s + batch_size, n)
        collection.upsert(
            ids=ids[s:e],
            documents=docs[s:e],
            metadatas=metadatas[s:e],
            embeddings=embeddings[s:e]
        )


# ============================================================
# CONTENT CANDIDATES (Chroma query)
# ============================================================

def content_candidates(collection, query_embedding, top_n=50):
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_n,
        include=["metadatas", "documents", "distances"]
    )

    out = []
    for md, dist in zip(res["metadatas"][0], res["distances"][0]):
        asin = md.get("asin")
        title = md.get("title", "")
        # cosine distance -> similarity-ish score
        score = float(1.0 - dist)
        score = max(0.0, min(1.0, score))
        out.append((asin, score, title))
    return out


# ============================================================
# CF CANDIDATES (co-purchase)
# ============================================================

def cf_candidates(cograph, asin, top_n=50):
    if asin not in cograph or not cograph[asin]:
        return []
    neighbors = cograph[asin]
    max_w = max(neighbors.values())
    out = []
    for b, w in neighbors.most_common(top_n):
        score = (w / max_w) if max_w > 0 else 0.0
        out.append((b, score))
    return out


# ============================================================
# HYBRID RECOMMENDATION (interactive)
# ============================================================

def resolve_query_to_asin(query, asin_to_title, title_to_asin):
    q = query.strip()
    if q in asin_to_title:
        return q, asin_to_title[q]

    qlow = q.lower()
    matches = [t for t in title_to_asin.keys() if qlow in t.lower()]
    if not matches:
        raise ValueError("No match found. Try an ASIN or longer title substring.")
    matches.sort(key=len)
    best_title = matches[0]
    return title_to_asin[best_title], best_title


def recommend_hybrid(collection, model, cograph, asin_to_title, title_to_asin,
                     query, k=10, alpha=0.6, content_top_n=60, cf_top_n=60):
    q_asin, q_title = resolve_query_to_asin(query, asin_to_title, title_to_asin)

    # Query embedding generated fresh per user query
    q_emb = model.encode([q_title], batch_size=1, show_progress_bar=False, convert_to_numpy=True)[0].tolist()

    cont = content_candidates(collection, q_emb, top_n=content_top_n)
    cont_scores = {a: s for a, s, _ in cont if a != q_asin}

    cf = cf_candidates(cograph, q_asin, top_n=cf_top_n)
    cf_scores = {a: s for a, s in cf if a != q_asin}

    union = set(cont_scores.keys()) | set(cf_scores.keys())
    scored = []
    for a in union:
        if a not in asin_to_title:
            continue
        cs = cont_scores.get(a, 0.0)
        fs = cf_scores.get(a, 0.0)
        final = alpha * cs + (1 - alpha) * fs
        scored.append((a, final, cs, fs, asin_to_title[a]))

    scored.sort(key=lambda x: x[1], reverse=True)
    return (q_asin, q_title), scored[:k]


# ============================================================
# EVALUATION (Hybrid vs Content only)
# Precision@K for multiple K and total batch runtime
#
# IMPORTANT CHANGE:
# - Hybrid timing now INCLUDES its own Chroma content query time
#   (no reusing 'cont' results from Content block).
# ============================================================

def precision_at_k(rec_asins, gt_set, k):
    if k <= 0:
        return 0.0
    topk = rec_asins[:k]
    return len(set(topk) & gt_set) / float(k)


def evaluate_hybrid_vs_content(collection, model, asin_list, asin_to_title, cograph,
                               alpha=0.6, ks=(10, 20, 30, 40, 50), num_queries=50):
    rng = random.Random(RANDOM_SEED)

    eligible = [a for a in asin_list if a in cograph and len(cograph[a]) > 0 and a in asin_to_title]
    if not eligible:
        raise ValueError("No eligible query items found (need co-purchase neighbors).")

    if len(eligible) < num_queries:
        num_queries = len(eligible)

    queries = rng.sample(eligible, num_queries)
    maxK = max(ks)

    prec_content = {k: [] for k in ks}
    prec_hybrid = {k: [] for k in ks}

    total_time_content = 0.0
    total_time_hybrid = 0.0

    for q_asin in tqdm(queries, desc="Evaluating (Content vs Hybrid)"):
        q_title = asin_to_title[q_asin]
        q_emb = model.encode([q_title], batch_size=1, show_progress_bar=False, convert_to_numpy=True)[0].tolist()
        gt = set(cograph[q_asin].keys())

        # -------------------------
        # Content-only (includes Chroma query)
        # -------------------------
        t0 = time.time()
        cont = content_candidates(collection, q_emb, top_n=maxK)
        cont_asins = [a for a, _, _ in cont if a != q_asin and a in asin_to_title]
        t1 = time.time()
        total_time_content += (t1 - t0)

        # -------------------------
        # Hybrid (includes its OWN Chroma query + CF + fusion)
        # -------------------------
        t0 = time.time()
        cont2 = content_candidates(collection, q_emb, top_n=maxK)  # do NOT reuse cont
        cont_scores = {a: s for a, s, _ in cont2 if a != q_asin}

        cf = cf_candidates(cograph, q_asin, top_n=maxK)
        cf_scores = {a: s for a, s in cf if a != q_asin}

        union = set(cont_scores.keys()) | set(cf_scores.keys())
        hyb = []
        for a in union:
            if a not in asin_to_title:
                continue
            cs = cont_scores.get(a, 0.0)
            fs = cf_scores.get(a, 0.0)
            final = alpha * cs + (1 - alpha) * fs
            hyb.append((a, final))

        hyb.sort(key=lambda x: x[1], reverse=True)
        hyb_asins = [a for a, _ in hyb]
        t1 = time.time()
        total_time_hybrid += (t1 - t0)

        # Precision@K
        for k in ks:
            prec_content[k].append(precision_at_k(cont_asins, gt, k))
            prec_hybrid[k].append(precision_at_k(hyb_asins, gt, k))

    return {
        "precision_content": {k: float(np.mean(prec_content[k])) for k in ks},
        "precision_hybrid": {k: float(np.mean(prec_hybrid[k])) for k in ks},
        "total_time_content": total_time_content,
        "total_time_hybrid": total_time_hybrid,
        "num_queries": num_queries
    }


def plot_precision_k(results, ks, out_path="precision_k.png"):
    c = [results["precision_content"][k] for k in ks]
    h = [results["precision_hybrid"][k] for k in ks]
    plt.figure()
    plt.plot(list(ks), c, marker="o", label="Content")
    plt.plot(list(ks), h, marker="o", label="Hybrid")
    plt.xlabel("K")
    plt.ylabel("Precision@K")
    plt.title("Precision@K vs K (Content vs Hybrid)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_runtime_total(results, out_path="runtime_total.png"):
    plt.figure()
    plt.bar(["Content", "Hybrid"],
            [results["total_time_content"], results["total_time_hybrid"]])
    plt.ylabel(f"Total time (s) for {results['num_queries']} queries")
    plt.title("Total Runtime Comparison (Content vs Hybrid)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def print_results(results, ks):
    print("\n=== Precision@K (Content vs Hybrid) ===")
    print("K\tContent\tHybrid")
    for k in ks:
        print(f"{k}\t{results['precision_content'][k]:.4f}\t{results['precision_hybrid'][k]:.4f}")

    print("\n=== Total time taken (same query batch) ===")
    print(f"Queries: {results['num_queries']}")
    print(f"Content total time: {results['total_time_content']:.4f} s")
    print(f"Hybrid  total time: {results['total_time_hybrid']:.4f} s")


# ============================================================
# MAIN
# ============================================================

def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Strongest guarantee: wipe persisted Chroma on every run
    if ALWAYS_FRESH_RUN:
        if os.path.exists(PERSIST_DIR):
            shutil.rmtree(PERSIST_DIR)

    print("\n1) Building random sample (reservoir sampling)...")
    sample = build_random_sample(DATA_PATH, SAMPLE_SIZE, RANDOM_SEED)
    print(f"Sample size: {len(sample)}")

    asin_to_title = {r["asin"]: r["title"] for r in sample if r["asin"] and r["title"]}
    title_to_asin = {r["title"]: r["asin"] for r in sample if r["asin"] and r["title"]}
    asin_list = list(asin_to_title.keys())

    print("\n2) Building co-purchase graph (within sampled ASINs only)...")
    cograph = build_copurchase_graph(sample)
    print(f"Co-purchase nodes: {len(cograph)}")

    print("\n3) Loading embedding model (CPU-only)...")
    model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")

    print("\n4) Creating ChromaDB collection (fresh)...")
    client = get_chroma_client(PERSIST_DIR)
    collection = recreate_collection(client, COLLECTION_NAME)
    print(" Rebuilt collection from scratch (fresh embeddings will be computed).")

    print("\n5) Preparing documents for embedding...")
    ids, docs, metas = [], [], []
    for r in sample:
        a, t = r["asin"], r["title"]
        if not a or not t:
            continue
        ids.append(a)
        docs.append(product_text(r))
        metas.append({"asin": a, "title": t, "group": r["group"]})

    print(f"\n6) Embedding {len(ids)} items on CPU (fresh embeddings every run)...")
    emb = model.encode(
        docs,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True
    ).astype(np.float32).tolist()

    print("\n7) Upserting embeddings into ChromaDB (batched)...")
    upsert_in_batches(collection, ids, docs, metas, emb, batch_size=UPSERT_BATCH)
    print(" ChromaDB populated (fresh run).")

    # --------------------------------------------------------
    # INTERACTIVE RECOMMENDATION
    # --------------------------------------------------------
    while True:
        q = input("\nEnter ASIN or part of product title (or 'eval' / 'exit'): ").strip()
        if q.lower() == "exit":
            return
        if q.lower() == "eval":
            break

        try:
            (q_asin, q_title), recs = recommend_hybrid(
                collection=collection,
                model=model,
                cograph=cograph,
                asin_to_title=asin_to_title,
                title_to_asin=title_to_asin,
                query=q,
                k=10,
                alpha=ALPHA,
                content_top_n=CONTENT_TOP_N,
                cf_top_n=CF_TOP_N
            )
            print(f"\nUsing query ASIN: {q_asin}")
            print(f"Title: {q_title}\n")
            print("Hybrid recommendations:")
            for i, (a, final, cs, fs, title) in enumerate(recs, 1):
                print(f"{i:2d}. {a} | final={final:.3f} (content={cs:.3f}, cf={fs:.3f}) | {title}")
        except Exception as e:
            print("Error:", e)

    # --------------------------------------------------------
    # EVALUATION + PLOTS (Hybrid vs Content)
    # --------------------------------------------------------
    print("\n=== Running evaluation: Precision@K (multi-K) + Total time ===")
    results = evaluate_hybrid_vs_content(
        collection=collection,
        model=model,
        asin_list=asin_list,
        asin_to_title=asin_to_title,
        cograph=cograph,
        alpha=ALPHA,
        ks=KS,
        num_queries=NUM_QUERIES
    )

    print_results(results, KS)
    plot_precision_k(results, KS, out_path="precision_k.png")
    plot_runtime_total(results, out_path="runtime_total.png")
    print("\nSaved plots: precision_k.png, runtime_total.png")


if __name__ == "__main__":
    main()
