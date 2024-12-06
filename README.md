# Streamlit Chatbot with Multilingual Support

This repository hosts a Streamlit-based chatbot application leveraging **LangChain** and **Ollama LLM** for conversational AI and [**IndicTrans Toolkit**](https://github.com/AI4Bharat/IndicTrans2/tree/main/huggingface_interface) for multilingual support. The chatbot includes dynamic conversation management, response streaming, and translation between Manipuri (Meitei Mayek) and English.

---

## Features

### Conversational AI:
- Built using **LangChain's ConversationChain** with a buffer memory to track chat history.
- Powered by **Ollama LLM** for generating context-aware responses.

### Multilingual Support:
- Seamless translation between English (`eng_Latn`) and Manipuri (`mni_Mtei`) using **IndicTrans Toolkit**.
- Enables natural and accessible communication in Manipuri.

### Streaming Responses:
- Simulates real-time response generation by streaming sentences with customizable delays.

### Custom Prompting:
- A user-friendly AI persona offering concise and helpful responses (less than 15 words).

### Dynamic Chat Interface:
- Persistent chat history displayed with an intuitive and visually appealing interface.

---

## Prerequisites

- **Python 3.8+**
- Install dependencies via `pip`:

    ```bash
    pip install streamlit langchain transformers IndicTransToolkit torch
    ```
---

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/streamlit-chatbot.git
cd streamlit-chatbot
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 2. Run the Application

- Enter your query in Manipuri.
- View responses in Manipuri.
