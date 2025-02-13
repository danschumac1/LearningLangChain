'''
2025-02-13
Author: Dan Schumacher
How to run:
   python ./src/memory.py
'''

import json
import openai
from utils.api import load_env, get_completion, LLM_MODEL
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import (
    ConversationBufferMemory, 
    ConversationBufferWindowMemory, 
    ConversationTokenBufferMemory,
    ConversationSummaryMemory,)

def main():
    #region SETUP
    load_env()
    openai.api_key = load_env()
    chat = ChatOpenAI(temperature=0.0, model=LLM_MODEL)
    memory = ConversationBufferMemory() # this is what stores the conversation
    conversation = ConversationChain(
        llm=chat,
        memory=memory,
        verbose=True
    )
    me1 = "Hi, my name is Dan."
    me2 = "What is my name?"
    rep1 = conversation.predict(input=me1)
    rep2 = conversation.predict(input=me2)
    print()
    print(me1, me2, rep1, rep2, sep='\n')
    # or you can do
    print(memory.buffer)
    print()

    memory = ConversationBufferMemory()
    memory.save_context({"input": "hi"}, {"output": "hello"}) # now the convo starts here

    # different types of memory
    # window memory (keeps just a windo) k = number of exchanges
    wind_memory = ConversationBufferWindowMemory(k=2) # this remembers just the last 2 exchanges

    # token memory (keeps just a windo) k = number of exchanges
    tok_memory = ConversationTokenBufferMemory(
        llm=chat,  # different llms have different ways of counting tokens
        max_token_limit=1000 # This remembers just the last 1000 tokens
        ) 
    
    # summary memory, uses an llm to summarize the conversation so far
    schedule = """There is a meeting at 8am with your product team.\
        You will need your powerpoint presentation prepared. \
        9am-12pm is blocked off for you to work on your project. \
        You have a lunch meeting with your manager at 12:30pm.\
        1pm-3pm is blocked off for you to work on your project. \
        3pm-5pm is blocked off for you to work on your project."""

    sum_memory = ConversationSummaryMemory(llm=chat, max_token_limit=1000)
    sum_memory.save_context({"input": "What is my schedule for today?"}, {"output": schedule})
    print('\n\n',sum_memory.load_memory_variables({}))
if __name__ == "__main__":
    main()



    #endregion
