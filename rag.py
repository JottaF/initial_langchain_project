from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vectordb import url_to_retriever
from main import llm

prompt = ChatPromptTemplate.from_template("""
{context}
Com base no contexto acima, responda: {input}
""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = url_to_retriever('https://pt.wikipedia.org/wiki/Oppenheimer_(filme)')
retriever_chain = create_retrieval_chain(retriever, document_chain)
response = retriever_chain.invoke({"input": "Qual o contexto do filme Oppenheimer, os personagens principais e os dados do seu lançamento?"})
print(response['answer'])