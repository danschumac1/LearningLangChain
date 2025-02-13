'''
2025-02-13
Author: Dan Schumacher
How to run:
   python ./src/chain.py
'''

import json
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import pandas as pd

from utils.api import load_env, LLM_MODEL as llm_model

def main():
    load_env()
    openai.api_key = load_env()
    chat = ChatOpenAI(temperature=0.0, model=llm_model)
    df = pd.read_csv("./data/Data.csv")

    prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe \
        a company that makes {product}?"
    )
    chain = LLMChain(llm=llm_model, prompt=prompt)
    product = "Queen Size Sheet Set"
    chain.run(product)
if __name__ == "__main__":
    main()