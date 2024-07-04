from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

dex_str = """Dexterity measures agility, reflexes, and balance."""
embedded_str = embeddings.embed_query(dex_str)
print("Stringa", dex_str)
print('\n----------------------------------------------------------------------------\n')
print("Embeddings:", embedded_str)
