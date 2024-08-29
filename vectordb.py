from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

def url_to_retriever(url):
  loader = WebBaseLoader(url)
  docs = loader.load()

  embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

  text_splitter = RecursiveCharacterTextSplitter()
  documents = text_splitter.split_documents(docs)
  vector = FAISS.from_documents(documents, embeddings)

  retriever = vector.as_retriever()
  return retriever