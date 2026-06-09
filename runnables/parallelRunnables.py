from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

# Initialize Mistral model
model = ChatMistralAI(model="mistral-small-2506")

# Converts AIMessage output into plain string
parser = StrOutputParser()

# Prompt for short explanation
short_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in 1-2 lines"
)

# Prompt for detailed explanation
detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in detail"
)

# Take separate topics from user
short_topic = input("Topic for short explanation: ")
detailed_topic = input("Topic for detailed explanation: ")

# RunnableParallel executes both chains simultaneously
chain = RunnableParallel({

    # Extracts value of "short" key from input dictionary
    "short": RunnableLambda(lambda x: x["short"])|short_prompt|model|parser,

    # Extracts value of "detailed" key from input dictionary
    "detailed": RunnableLambda(lambda x: x["detailed"])|detailed_prompt|model|parser
})

# Input for both parallel branches
result = chain.invoke({
    "short": {"topic": short_topic},
    "detailed": {"topic": detailed_topic}
})

# Display short explanation
print("\nShort Explanation:")
print(result["short"])

# Display detailed explanation
print("\nDetailed Explanation:")
print(result["detailed"])