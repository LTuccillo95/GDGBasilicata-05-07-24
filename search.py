from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings

# Carico il modello di embedding
embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


#Dichiaro il mio vectorDB
vectordb = Chroma(embedding_function=embeddings, persist_directory='./chromaDB', collection_name="src_5e_spell")

# Ricerco
import pprint

pprint.pp(vectordb.similarity_search('shield'))
print('------------------------------------------------------------------------------------------------')
pprint.pp(vectordb.similarity_search_by_vector(embeddings.embed_query('shield')))
