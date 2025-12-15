### Hybrid Recommendation System using Vector Embeddings and Co-Purchase Collaborative Filtering

This project implements a hybrid recommendation system that combines content-based semantic similarity (via vector embeddings) with collaborative filtering using co-purchase signals, powered by a vector database (ChromaDB) and HNSW approximate nearest neighbor search.

The system is designed to improve recommendation accuracy, diversity, and robustness, addressing limitations of standalone content-based or collaborative approaches.

⸻

 Project Overview

Traditional recommender systems typically rely on either:
- Content-based filtering (semantic similarity of item descriptions), or
- Collaborative filtering (user behavior such as co-purchases).

Each approach has limitations:
- Content-based systems often produce repetitive or near-duplicate recommendations.
- Collaborative systems suffer from sparsity and cold-start issues.

This project proposes a hybrid architecture that:
- Uses Sentence Transformer embeddings for semantic similarity.
- Uses co-purchase graphs as implicit behavioral signals.
- Combines both using a tunable hybrid weighting factor.

⸻

Key Features
- Sentence-BERT embeddings (all-MiniLM-L12-v2, 384 dimensions)
- Vector database (ChromaDB) for fast semantic retrieval
- HNSW ANN indexing for efficient nearest neighbor search
- Item-item collaborative filtering using co-purchase graphs
- Hybrid score fusion with adjustable parameter \alpha
- Precision@K evaluation across multiple K values
- Reproducible experimental setup
⸻

Dataset
- Amazon Product Metadata Dataset
- ~542,684 products
- Fields used:
- ASIN
- Product title
- Category / group
- Similar products (co-purchase list)
Due to hardware constraints, experiments are conducted on a random subset (e.g., 100K products) while preserving valid co-purchase relationships.

Implementaion Steps:

1.Downloading the dependencies (requirement.txt)

2.Link to download the data set -https://snap.stanford.edu/data/amazon-meta.html

3. Run the file "Hybrid_Recom_vecdb.py"

Output:
<img width="1437" height="470" alt="image" src="https://github.com/user-attachments/assets/0450ba44-14e6-4859-a865-c47872c07371" />

<img width="1423" height="817" alt="image" src="https://github.com/user-attachments/assets/47a991ae-a51b-405a-9412-72fa6815aa3c" />



Evaluation Methodology

Ground Truth Proxy

•	Explicit ratings are unavailable.

•	Co-purchase neighbors are treated as relevant items.

Metrics

- Precision@K, where:

$$ \text{Precision@K} = \frac{|R_K \cap G|}{K} $$

- Evaluated for: $$ K \in \{10, 20, 30, 40, 50\} $$

-Total time taken- The sum of execution time for all recommendation queries in a test batch for both the models.

Compared Models:
- Content-only (vector similarity)
- Hybrid (content + co-purchase)

⸻

Limitations
- Increased computation due to dual retrieval paths
- Sensitivity to hybrid weight \alpha
- Local ChromaDB is not fully distributed
- Cold-start remains a challenge for sparse items

⸻

Future Work
- Cold-start mitigation strategies
- Cloud-based or distributed vector databases
- GPU-accelerated embedding generation
- Learning \alpha dynamically using validation data
- Incorporating user-level personalization


⸻
Authors
- Karthikraj Donthula
  MS in CSDS, Case Western Reserve University
  
- Athish Vikraman Periyapalayam Ramesh
  ECSE, Case Western Reserve University
