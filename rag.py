from langchain_community.vectorstores.chroma import Chroma
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_google_vertexai.chat_models import ChatVertexAI

## Data Source https://github.com/5e-bits/5e-database

embeddings = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

#Inizializo il mio LLM
llm = ChatVertexAI(model_name='gemini-pro')

vectordb = Chroma(embedding_function=embeddings, persist_directory='./chromaDB', collection_name="src_5e_spell")


chain_conversational = ConversationalRetrievalChain.from_llm(llm, vectordb.as_retriever(), verbose=True)

chain_retrvie_source = RetrievalQA.from_chain_type(llm=llm, retriever=vectordb.as_retriever(), verbose=True, return_source_documents=True)


# print(chain_conversational.invoke({'question': 'what shield does?', 'chat_history': ''}))
# print(chain_conversational.invoke({'question': 'what the spell shield does?', 'chat_history': ''}))
# print(chain_conversational.invoke({'question': 'what aid does?', 'chat_history': ''}))

# print(chain_retrvie_source.invoke({'query': 'what shield does?', 'chat_history': ''}))
# print(chain_retrvie_source.invoke({'query': 'what the spell shield does?', 'chat_history': ''}))
#print(chain_retrvie_source.invoke({'query': 'what aid does?', 'chat_history': ''}))
