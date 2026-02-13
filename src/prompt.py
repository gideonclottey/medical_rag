# Now let bind the two, the retriever and then llm
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

system_prompt =(
    "You are an Medical assistant for question-answering tasks."
    "use the following pieces of retrieved context to answer"
    "the question. if you don't know the answer, say that you"
    "dont't know. Use three sentences and keep the answer concise"
    "\n\n"
    "{context}"

)
