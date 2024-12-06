
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
import requests
import time
import re
from IndicTransToolkit import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Title on the page
st.markdown(
    "<h2 style='text-align: center; color: #4CAF50; font-family: Arial;'>ꯆꯦꯠ ꯖꯤ.ꯄꯤ.ꯇꯤ </h2>",
    unsafe_allow_html=True,
)

template = """
You are a helpful and friendly AI assistant. You aim to provide concise responses of less than 15 words.

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:
"""

PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)

chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant:"),
    llm=Ollama(model="llama3.2"),
)

# Initialize the message history with an AI-generated greeting if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Get the AI's first message (a greeting)

    # first_message = chain.predict(input="Hello!")
    # st.session_state.messages.append({"role": "assistant", "content": first_message})
    # first_message = chain.predict(input="Hello!")
    st.session_state.messages.append({"role": "assistant", "content": "ꯈꯨꯔꯨꯃꯖꯔꯤ, ꯉꯁꯤ ꯃꯇꯧ ꯀꯔꯝꯅꯥ ꯂꯩꯔꯤꯕꯒꯦ?"})

# Display the chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from IndicTransToolkit import IndicProcessor

# Determine the device

ip = IndicProcessor(inference=True)

def translate(input_sentences, src_lang, tgt_lang):
    print("Source Language:",src_lang)
    print("Target Language:",tgt_lang)

    if src_lang == "eng_Latn":
        # Model and tokenizer initialization
        model_name = "ai4bharat/indictrans2-en-indic-1B"
    elif src_lang == "mni_Mtei":
        model_name = "ai4bharat/indictrans2-indic-en-1B"
        
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)

    DEVICE = "cpu"
    # Move model to the appropriate device
    model.to(DEVICE)

    # Preprocess the batch
    batch = ip.preprocess_batch(
        input_sentences,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    # Tokenize the sentences and generate input encodings
    inputs = tokenizer(
        batch,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        return_attention_mask=True,
    ).to(DEVICE)

    # Generate translations using the model
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            use_cache=True,
            min_length=0,
            max_length=256,
            num_beams=5,
            num_return_sequences=1,
        )

    # Decode the generated tokens into text
    with tokenizer.as_target_tokenizer():
        generated_tokens = tokenizer.batch_decode(
            generated_tokens.detach().cpu().tolist(),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    # Postprocess the translations, including entity replacement
    translations = ip.postprocess_batch(generated_tokens, lang=tgt_lang)

    # Return the translations
    return translations[0]

# Function to simulate streaming the assistant's response sentence by sentence
def stream_response(response: str, delay: float = 0.5):
    """Simulate streaming response by sentences with a delay."""
    # Split the response into sentences using regex
    sentences = re.split(r'(?<=[.!?]) +', response)
    
    # Stream each sentence with a delay, appending to the same line
    full_response = ""
    response_container = st.empty()  # Create a placeholder for the response

    for sentence in sentences:
        full_response += sentence + " "  # Add the sentence to the ongoing response
        response_container.markdown(full_response, unsafe_allow_html=True)  # Update the placeholder with the ongoing response
        time.sleep(delay)

# Taking user input
if user_input := st.chat_input("What is up?"):
    print("Input received",user_input)
    # Adding user message to session state
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    try:
        print("Calling translate function")
        translated_input = translate([user_input], src_lang="mni_Mtei", tgt_lang="eng_Latn")
        print("Translated user input:",translated_input)
        # console.print(f"[blue]Translated: {translated_text}")
    except Exception as e:
        print(e)
        # console.print(f"[red]Translation Error: {str(e)}")

    # Generating the assistant's response using the chain
    with st.chat_message("assistant"):
        response = chain.predict(input=translated_input)
        print("Generated Response by LLM:",response)
        try:
            response = translate([response], src_lang="eng_Latn", tgt_lang="mni_Mtei")
            # console.print(f"[blue]Translated: {translated_text}")
        except Exception as e:
            print(e)
    
        # Stream the response sentence by sentence
        stream_response(response)

    # Adding the assistant message to session state
    st.session_state.messages.append({"role": "assistant", "content": response})
