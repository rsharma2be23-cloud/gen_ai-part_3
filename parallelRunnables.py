from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

short_prompt = ChatPromptTemplate.from_template(
    "Explain {short_topic} in 1-2 lines"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {detailed_topic} in detail"
)

short_topic = input("Topic for short explanation: ")
detailed_topic = input("Topic for detailed explanation: ")

chain = RunnableParallel({
    "short": short_prompt | model | parser,
    "detailed": detailed_prompt | model | parser
})

result = chain.invoke({
    "short_topic": short_topic,
    "detailed_topic": detailed_topic
})

print("\nShort Explanation:")
print(result["short"])

print("\nDetailed Explanation:")
print(result["detailed"])