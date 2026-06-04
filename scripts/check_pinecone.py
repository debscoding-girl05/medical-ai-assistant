"""Diagnostic: inspect the Pinecone 'medical-chatbot' index.

Run locally (with your .env containing PINECONE_API_KEY):
    python scripts/check_pinecone.py

It prints the index dimension, total vector count, and lists indexes, so we can
decide whether the chatbot can switch to OpenAI embeddings (1536-dim) or must
keep the existing dimension.
"""
import os
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "medical-chatbot"


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("❌ PINECONE_API_KEY not found in environment/.env")
        return

    try:
        from pinecone import Pinecone
    except ImportError:
        print("❌ pinecone package not installed. Run: pip install pinecone")
        return

    pc = Pinecone(api_key=api_key)

    print("=== Indexes in this Pinecone project ===")
    names = [idx["name"] for idx in pc.list_indexes()]
    print(names or "(none)")

    if INDEX_NAME not in names:
        print(f"\n❌ Index '{INDEX_NAME}' does NOT exist in this project/key.")
        print("   -> The chatbot needs the index created and populated.")
        return

    desc = pc.describe_index(INDEX_NAME)
    print(f"\n=== Index '{INDEX_NAME}' ===")
    print(f"dimension : {desc['dimension']}")
    print(f"metric    : {desc['metric']}")
    print(f"host      : {desc.get('host')}")

    index = pc.Index(INDEX_NAME)
    stats = index.describe_index_stats()
    total = stats.get("total_vector_count", 0)
    print(f"vectors   : {total}")

    print("\n=== Verdict ===")
    if total == 0:
        print("⚠️  Index exists but is EMPTY — it must be populated before the chatbot works.")
    elif desc["dimension"] == 384:
        print("ℹ️  384-dim (MiniLM). To go torch-free with OpenAI embeddings (1536-dim),")
        print("    this index must be recreated at dimension 1536 and re-populated.")
    elif desc["dimension"] == 1536:
        print("✅ 1536-dim — already compatible with OpenAI embeddings.")
    else:
        print(f"ℹ️  dimension {desc['dimension']} — match your embedding model to this.")


if __name__ == "__main__":
    main()
