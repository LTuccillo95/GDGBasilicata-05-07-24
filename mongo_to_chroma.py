from langchain_core.documents import Document
from langchain.document_loaders import MongodbLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = MongodbLoader(
    connection_string="mongodb://localhost:27017/",
    db_name='admin',
    collection_name='src-spell',
)

# loader = MongodbLoader(
#     connection_string="mongodb://localhost:27017/",
#     db_name='admin',
#     collection_name='src-attribute',
# )

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=128, chunk_overlap=32)
splitted_docs = text_splitter.split_documents(docs)

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# vectorDB = Chroma.from_documents(docs, embeddings, persist_directory='./chromaDB', collection_name="src_5e_splitted_spell")
vectorDB = Chroma.from_documents(splitted_docs, embeddings, persist_directory='./chromaDB', collection_name="src_5e_splitted_spell")
vectorDB.persist()