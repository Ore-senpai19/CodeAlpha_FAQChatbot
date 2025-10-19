import streamlit as st
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from faq_data import faq_pairs

if "memory" not in st.session_state:
    st.session_state.memory = {
        "name": None,
        "info": {}
    }

# Download NLTK resources (first run only)
nltk.download("punkt")
nltk.download("stopwords")

# Data Preparation
questions = [q for q, a in faq_pairs]
answers = [a for q, a in faq_pairs]

# Vectorize questions (ignore filler words)
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(questions)

# Detect if message looks like Pidgin / Naija English
def is_pidgin(text):
    pidgin_keywords = [
        "wetin", "dey", "abeg", "omo", "wahala", "sabi", "fit",
        "no vex", "sharp", "e choke", "gist", "how far", "kampe",
        "beta", "no wahala", "my gee", "my guy", "you dey", "gbas",
        "na", "dem", "go", "make am"
    ]
    text_lower = text.lower()
    return any(word in text_lower for word in pidgin_keywords)

# Chatbot Logic with Confidence Threshold
def get_answer(user_query):
    user_vec = vectorizer.transform([user_query])
    similarity = cosine_similarity(user_vec, X)
    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score < 0.3:
        # Add a friendly fallback
        if is_pidgin(user_query):
            return "I no too sure about dat one oh 😅. Fit make you rephrase am?"
        else:
            return "Hmm, I’m not sure about that yet 🤔 — could you rephrase your question?"

    answer = answers[best_match]

    # If user spoke Pidgin, reply in lighter Naija tone
    if is_pidgin(user_query):
        if not answer.endswith("😂") and not answer.endswith("😅"):
            answer = answer + " 😅"
    return answer

def update_memory(user_input):
    # Detect and remember name
    if "my name is" in user_input.lower():
        name = user_input.lower().split("my name is")[-1].strip().capitalize()
        st.session_state.memory["name"] = name
        return f"Nice to meet you, {name}! I'll remember your name 😎."

    # Save extra info if the user says "I am" or "I'm"
    if "i am" in user_input.lower() or "i'm" in user_input.lower():
        parts = user_input.split()
        try:
            index = parts.index("am") if "am" in parts else parts.index("I'm")
            info = " ".join(parts[index+1:])
            st.session_state.memory["info"]["about"] = info
            return f"Okay, I have noted your name is {info} 😎."
        except:
            pass
        if st.session_state.memory["name"]:
            answer = answer.replace("you", f"{st.session_state.memory['name']}")

    return None

# Streamlit Interface
st.set_page_config(page_title="FAQ Chatbot", page_icon="💬", layout="centered")

st.title("💬 Friendly FAQ Chatbot")
st.caption("Ask me anything — from daily topics to tech, education, or even fun facts!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hey there 👋 I'm your friendly chatbot! What would you like to know today?"}
    ]

# Display chat history using Streamlit’s chat UI
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_input := st.chat_input("Ask anything..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Check if the message should be remembered (name, info, etc.)
    memory_response = update_memory(user_input)

    # Decide what the bot should reply
    if memory_response:
        response = memory_response
    else:
        response = get_answer(user_input)

    # Add bot message to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display both messages
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        st.write(response)
