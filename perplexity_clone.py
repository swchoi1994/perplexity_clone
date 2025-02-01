import os
import streamlit as st
from tavily import TavilyClient
from groq import Groq
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize APIs
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Configure Streamlit page
st.set_page_config(
    page_title="새로운 검색의 시작",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# User database (For demo purposes - use proper database in production)
VALID_USERS = {
    "swchoi1994": "password123!",
    "user2": "securePass456@",
    "user3": "llama3$789"
}

# Custom CSS for dark mode
def set_dark_mode():
    st.markdown("""
    <style>
        .stApp {
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .stTextInput input, .stTextInput textarea {
            background-color: #2d2d2d !important;
            color: white !important;
        }
        .stButton button {
            background-color: #4CAF50 !important;
            color: white !important;
        }
        .warning {
            color: #ff4b4b !important;
            font-size: 0.9em;
        }
    </style>
    """, unsafe_allow_html=True)

# Enhanced authentication system
def authenticate():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.current_user = None

    if not st.session_state.authenticated:
        st.title("User Login 🔒")
        
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username in VALID_USERS and VALID_USERS[username] == password:
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.rerun()
            else:
                st.error("Invalid username or password")
        st.stop()

def main():
    set_dark_mode()
    authenticate()
    
    # Initialize conversation memory
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    st.title(f"Welcome {st.session_state.current_user}")
    
    # Display chat history
    for msg in st.session_state.memory.buffer:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg.content)
    
    # Search interface
    query = st.chat_input("Ask anything...")
    
    if query:
        # Add user message to memory and display
        st.session_state.memory.chat_memory.add_user_message(query)
        with st.chat_message("user"):
            st.write(query)
        
        with st.status("🔍 Processing...", expanded=True) as status:
            # Create context-aware prompt
            prompt_template = ChatPromptTemplate.from_template(
                """You are an AI research assistant. Consider our conversation history:
                {chat_history}
                
                New Query: {query}
                
                Search Results: {search_results}
                
                Provide a comprehensive answer using the context and previous conversation."""
            )
            
            # Web search
            search_result = tavily.search(
                query=query,
                search_depth="advanced",
                include_answer=True
            )
            
            # Format prompt with memory
            formatted_prompt = prompt_template.format(
                chat_history=st.session_state.memory.buffer,
                query=query,
                search_results=search_result['answer']
            )
            
            # Generate response
            response = groq_client.chat.completions.create(
                messages=[{"role": "user", "content": formatted_prompt}],
                model="deepseek-r1-distill-llama-70b",
                temperature=0.6,
                max_completion_tokens=131072,
                top_p=0.95
            )
            
            # Add response to memory
            ai_response = response.choices[0].message.content
            st.session_state.memory.chat_memory.add_ai_message(ai_response)
            
            status.update(label="✅ Response Ready", state="complete")
        
        # Display AI response
        with st.chat_message("assistant"):
            st.write(ai_response)
            
            with st.expander("View Sources"):
                for result in search_result['results']:
                    st.markdown(f"""
                    **{result['title']}**  
                    URL: {result['url']}  
                    Content: {result['content']}
                    """)
                    st.divider()

    # Add clear memory button
    if st.button("Clear Conversation History"):
        st.session_state.memory.clear()
        st.rerun()

if __name__ == "__main__":
    main()