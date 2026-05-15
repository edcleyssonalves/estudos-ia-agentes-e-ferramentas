import os
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

os.environ['OPENAI_API_KEY']

model = ChatOpenAI(
    model='gpt-3.5-turbo',
    max_completion_tokens=500,

)

api_wrapper = WikipediaAPIWrapper(
    lang='pt',
    top_k_results=1,
    doc_content_chars_max=3000
)

wikipedia_tool = WikipediaQueryRun(
    api_wrapper=api_wrapper
)

agent_executor = create_python_agent(
    llm=model,
    tool=wikipedia_tool,
    verbose=True,
)

prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
    Pesquise na web sobre {query} e forneça um resumo sobre o assunto.
    Responda tudo em português brasileiro.
    '''
)

query = 'Alan Turing'
prompt = prompt_template.format(query=query)

response = agent_executor.invoke(prompt)

print(response.get('output'))