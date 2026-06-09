from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# Initialize Mistral model
model = ChatMistralAI(model="mistral-small-2506")

# Converts AIMessage output into plain string
parser = StrOutputParser()

code_prompt=ChatPromptTemplate.from_messages({
  ("system","you are a code generator"),
  ("human","{topic}")
})
explain_prompt=ChatPromptTemplate.from_messages([
  ("system","you explain the code in simple language:\n{code}"),
  ("human","explain the following code in simple language")
])

seq=code_prompt|model|parser

seq2=RunnableParallel(
  {"answer_code": RunnablePassthrough(),
   "explanation":explain_prompt|model|parser

  }
)

chain=seq|seq2

result=chain.invoke({"topic": "please write a codeof binary sort "})


print(result['answer_code'])
print(result['explanation'])