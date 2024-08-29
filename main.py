from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts  import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import sys

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

def oscar(film, year):
  prompt_oscar = PromptTemplate.from_template("Quantos oscars o filme {film} ganhou em {year}?")

  oscar_chain = prompt_oscar | llm

  response = oscar_chain.invoke({'film': film, 'year': year})

  return response

if __name__=="__main__":
  response = oscar('Oppenheimer', 2024)
  print(response)