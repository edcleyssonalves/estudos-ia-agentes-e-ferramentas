from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


api_wrapper = WikipediaAPIWrapper(
    lang='pt'
)

wikipedia = WikipediaQueryRun(
    api_wrapper=api_wrapper
)

resultado = wikipedia.run('Quem foi Alan Turing?')
print(resultado)
