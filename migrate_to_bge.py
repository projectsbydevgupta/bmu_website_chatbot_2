"""
BMU Knowledge Base Migration to BAAI/bge-large-en-v1.5 (1,024 Dimensions)
========================================================================
Extracts all existing documents from 'bmu_summaries' (384-d),
computes normalized 1,024-dim dense embeddings using BAAI/bge-large-en-v1.5,
and indexes them into the new ChromaDB Cloud collection 'bmu_bge_knowledge'.
"""

import os
import sys
import time
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

SOURCE_COLLECTION = "bmu_summaries"
TARGET_COLLECTION = "bmu_bge_knowledge"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64

def main():
    print("=" * 60)
    print(f"Migrating to 1,024-d Model: {MODEL_NAME}")
    print(f"Source: {SOURCE_COLLECTION} -> Target: {TARGET_COLLECTION}")
    print("=" * 60)

    # 1. Connect to ChromaDB Cloud
    client = chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )
    src_col = client.get_collection(name=SOURCE_COLLECTION)
    total_docs = src_col.count()
    print(f"Found {total_docs} documents in source collection '{SOURCE_COLLECTION}'.")

    # 2. Get or create target collection
    target_col = client.get_or_create_collection(
        name=TARGET_COLLECTION,
        metadata={"hnsw:space": "cosine", "model": MODEL_NAME, "dimension": 1024}
    )
    existing_count = target_col.count()
    print(f"Target collection '{TARGET_COLLECTION}' currently has {existing_count} documents.")

    if existing_count >= total_docs:
        print(f"Target collection is already fully populated ({existing_count}/{total_docs}). Migration not needed!")
        return

    # 3. Load BGE Model
    print(f"\nLoading SentenceTransformer: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded successfully (1,024 dimensions).")

    # 4. Fetch all documents from source in chunks
    print("\nFetching source documents...")
    all_ids = []
    all_docs = []
    all_metas = []

    # Read all data with pagination (ChromaDB Cloud has a limit <= 300 per get request)
    offset = 0
    fetch_chunk = 200
    while offset < total_docs:
        batch = src_col.get(
            limit=fetch_chunk,
            offset=offset,
            include=["documents", "metadatas"]
        )
        b_ids = batch.get("ids", [])
        if not b_ids:
            break
        all_ids.extend(b_ids)
        all_docs.extend(batch.get("documents", []))
        all_metas.extend(batch.get("metadatas", []))
        offset += len(b_ids)
        print(f"  Retrieved {offset}/{total_docs} documents...")

    print(f"Total retrieved: {len(all_docs)} documents.")

    # Clean metadatas (ensure no None or empty values)
    cleaned_metas = []
    for m in all_metas:
        meta_dict = m if isinstance(m, dict) else {}
        clean_dict = {}
        for k, v in meta_dict.items():
            if v is not None:
                clean_dict[k] = v
            else:
                clean_dict[k] = ""
        cleaned_metas.append(clean_dict)

    # 5. Embed and Upsert in batches
    print(f"\nEmbedding & Upserting into '{TARGET_COLLECTION}' in batches of {BATCH_SIZE}...")
    start_time = time.time()

    for i in range(0, len(all_docs), BATCH_SIZE):
        batch_ids = all_ids[i:i + BATCH_SIZE]
        batch_docs = all_docs[i:i + BATCH_SIZE]
        batch_metas = cleaned_metas[i:i + BATCH_SIZE]

        # BGE embeddings with cosine normalization
        batch_embs = model.encode(
            batch_docs,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

        target_col.upsert(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_metas,
            embeddings=batch_embs,
        )
        print(f"  Indexed {min(i + BATCH_SIZE, len(all_docs))}/{len(all_docs)} documents...", end="\r", flush=True)

    elapsed = round(time.time() - start_time, 2)
    final_count = target_col.count()
    print(f"\n\nMigration complete in {elapsed}s! Target collection now contains {final_count} documents.")

    # 6. Verification query
    test_query = "What B.Tech specializations are offered at BMU?"
    q_emb = model.encode([f"Represent this sentence for searching relevant passages: {test_query}"], normalize_embeddings=True).tolist()
    results = target_col.query(query_embeddings=q_emb, n_results=3)
    print("\nVerification Retrieval Test:")
    print(f"Query: {test_query}")
    for idx, (doc, meta, dist) in enumerate(zip(results["documents"][0], results["metadatas"][0], results["distances"][0])):
        sim = round(max(0.0, 1.0 - dist), 4)
        print(f" [{idx+1}] Score: {sim} | URL: {meta.get('url', 'N/A')}")
        print(f"     Snippet: {doc[:120]}...\n")

if __name__ == "__main__":
    main()
