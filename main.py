from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts  import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os
import sys

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

def generate_name(animal_type):
  prompt_animal_name = PromptTemplate(
    input_variables=['animal_type'],
    template="Me sugira 5 nomes para {animal_type} filhotes."
  )

  animal_name_chain = LLMChain(llm=llm, prompt=prompt_animal_name)
  response = animal_name_chain({'animal_type': animal_type})
  return response['text']

if __name__=="__main__":
  print(generate_name('tanajuras'))