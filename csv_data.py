from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain_google_vertexai.chat_models import ChatVertexAI
import pandas as pd
import os

# Dichiaro l' LLM
llm = ChatVertexAI(model_name='gemini-pro')

# Carico il .csv
df = pd.read_csv("csv_example.csv", sep=';')
df = df.drop(axis='columns', columns=['numero protocollo', 'ufficio competenza', 'stato'])

# Creo il mio agent
agent = create_pandas_dataframe_agent(llm, df, verbose=True, allow_dangerous_code=True, handle_parsing_errors=True)

# Provo il mio agent

# print(agent.invoke({'input': "Qual è la fattura con l'importo imponibile più alto"}))

# path = os.path.join(os.path.dirname(__file__), 'graph.png')
# print(agent.invoke({'input': f'Fammi il grafico a torta dei beneficiari per le fatture ricevute e salvalo sul file system a questo path {path}, non mostrarmelo'}))