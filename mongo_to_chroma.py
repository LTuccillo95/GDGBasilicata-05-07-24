from langchain_core.documents import Document
from langchain.document_loaders import MongodbLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
loader = MongodbLoader(
    connection_string="mongodb://localhost:27017/",
    db_name='admin',
    collection_name='src-attribute',
)

docs = loader.load()

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

vectorDB = Chroma.from_documents(docs, embeddings, persist_directory='./chromaDB', collection_name="src_5e_attribute")
vectorDB.persist()