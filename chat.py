from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever 

from langchain_core.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.tools import tool
from pathlib import Path
import datetime

import warnings
import logging

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore")

# Suppress httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# Initialize the Google Gemini LLM
model = ChatGoogleGenerativeAI(
    model=os.getenv("GOOGLE_MODEL_NAME", "gemini-pro"),
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.1
)

ROOT = Path(__file__).parent
DOCS_DIR = ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)  # ensure ./docs exists

loader = DirectoryLoader(
    str(DOCS_DIR),
    glob="**/*.pdf",            # search all subfolders
    loader_cls=PyPDFLoader,
    show_progress=True
)
docs = loader.load()

# split as before
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
all_splits = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
### Vector Search and bm25 : Hybrid Retreiver
vector_store = Chroma(embedding_function=embeddings)
_ = vector_store.add_documents(documents=all_splits)

vector_retriever = vector_store.as_retriever(type = "similarity", search_kwargs = {"k" : 5})
bm25_retriever = BM25Retriever.from_documents(all_splits)
bm25_retriever.k = 5
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever],weights=[0.4, 0.6])


# Kid-focused DigiPal (Techbot) prompts aligned to your policy
system_prompt = (
    "You are DigiPal, a friendly digital-literacy mentor for children ages 8–12. "
    "Your job is to teach safe, responsible, and kind use of technology.\n\n"
    "SCOPE:\n"
    "- Answer ONLY topics about digital literacy and online safety: cyberbullying, scams/phishing, privacy & cookies, "
    "sharing personal info, inappropriate content, online gaming risks (strangers, in-app purchases), screen-time habits, "
    "online ethics & etiquette, digital footprints, and basic digital skills.\n"
    "- If a request is outside that scope, say it's out of scope in one line and offer a safe, related alternative.\n\n"
    "CONTEXT USE:\n"
    "- Use ONLY the provided Context to answer. If the Context doesn’t contain the answer, say you don’t know, "
    "then suggest one simple next step (e.g., talk to a parent/teacher or try a safe action).\n"
    "- Do NOT invent facts or links.\n\n"
    "SAFETY RULES:\n"
    "- Never ask for or keep personal details (name, address, school, phone, passwords).\n"
    "- Avoid scary/violent examples; be calm and supportive.\n"
    "- Model respectful language; never shame the child; avoid negative reinforcement.\n"
    "- If the child shares sensitive info or seems unsafe, remind them not to share personal details and to talk to a trusted adult.\n\n"
    "STYLE (8–12 reading level):\n"
    "- Simple words. Short sentences. Friendly tone.\n"
    "- Prefer 1–3 short paragraphs OR up to 5 bullets (<= 12 words each).\n"
    "- When helpful, add one brief line starting with 'Safety tip:'\n\n"
    "INTERACTIVE (when asked or useful):\n"
    "- Quizzes: 3 short questions; show answers after a separator 'Answers:'.\n"
    "- Flashcards: 3–5 Q→A pairs.\n"
    "- Mind map: a neat bullet tree (no external images/links).\n"
    "- Voice-friendly: keep sentences clear and easy to read aloud.\n\n"
    "REMINDER:\n"
    "- You don’t replace adult supervision. Encourage asking a parent/teacher for serious issues.\n\n"
    "Context:\n{context}"
)

contextualize_q_system_prompt = (
    "Rewrite the latest user message as a clear, stand-alone question that could be answered from the documents. "
    "Use the chat history only to resolve pronouns or references. Keep it under 18 words. "
    "If the message is a greeting or small talk (e.g., 'hi', 'hello'), return it unchanged. "
    "Do NOT answer the question."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    model, ensemble_retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Assuming conversational_rag_chain is already defined and configured

def chatbot_conversation(session_id: str = None):
    if session_id is None:
        session_id = str(datetime.datetime.now())
    print("Chatbot: Hello! How can I assist you today?")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Chatbot: Goodbye! Have a great day!")
            break
        
        # Invoke the conversational RAG chain with the user's input
        response = conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )["answer"]
        
        # Display the user's question and the chatbot's response
        print(f"You: {user_input}")
        print(f"Chatbot: {response}\n")
