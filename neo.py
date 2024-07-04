from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_google_vertexai.chat_models import ChatVertexAI  

## Esempio di Query Cypher

# MATCH (m:Movie)-[:IN_GENRE]->(g:Genre)
# WHERE g.name = 'Comedy'
# RETURN m.title


# Mi Collego la graphdb
graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="neo4jneo4j", enhanced_schema=True)

# Creo la mia chain/agent
chain = GraphCypherQAChain.from_llm(
    ChatVertexAI(temperature=0), graph=graph, verbose=True
)

# Provo la mia chain
print(chain.invoke({"query": "Give me a list of the comedy movie?"}))