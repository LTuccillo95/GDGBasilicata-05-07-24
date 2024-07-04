from langchain_google_vertexai.chat_models import ChatVertexAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent


#Inizializo il mio LLM
llm = ChatVertexAI(model_name='gemini-pro')

#Stabilisco una connessione al DB
db = SQLDatabase.from_uri("mysql+mysqlconnector://root:root@localhost/gdg")

#Creo il mio Agent
sql_agent = create_sql_agent(llm=llm, db=db, verbose=True)


#Prompt

## PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
## FORMAT_INSTRUCTIONS = """Use the following format:
## 
## Question: the input question you must answer
## Thought: you should always think about what to do
## Action: the action to take, should be one of [{tool_names}]
## Action Input: the input to the action
## Observation: the result of the action
## ... (this Thought/Action/Action Input/Observation can repeat N times)
## Thought: I now know the final answer
## Final Answer: the final answer to the original input question"""
## SUFFIX = """Begin!
## 
## Question: {input}
## Thought:{agent_scratchpad}"""



#Provo il mio Agent SQL

#print(sql_agent.invoke("What is the highest importo totale from gdg table?"))

#print(sql_agent.invoke("Scrivimi la query per ottenere l'importo totale più alto su questo db. Restituiscimi solo la query senza altre spiegazioni"))
