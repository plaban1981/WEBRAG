import streamlit as st
from rag_app import WebRAG
import time

# Set page configuration
st.set_page_config(
    page_title="Web RAG Assistant",
    page_icon="🌐",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-container {
        border-radius: 10px;
        padding: 20px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .user-message {
        background-color: #2e7bf3;
        color: white;
        padding: 15px;
        border-radius: 15px;
        margin: 5px 0;
    }
    .assistant-message {
        background-color: #white;
        padding: 15px;
        border-radius: 15px;
        margin: 5px 0;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = WebRAG()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'url_processed' not in st.session_state:
    st.session_state.url_processed = False
if 'current_url' not in st.session_state:
    st.session_state.current_url = ""

# Function to reset chat history
def reset_chat_history():
    st.session_state.chat_history = []
    st.session_state.current_url = ""

# Header
st.title("🌐 Web RAG Assistant")
st.markdown("### Ask questions about any webpage")

# Sidebar
with st.sidebar:
    st.header("Settings")
    url = st.text_input("Enter webpage URL:")
    
    # Add scraping method selection
    scraping_method = st.selectbox(
        "Select Scraping Method",
        ["beautifulsoup", "scrapegraph", "crawl4ai"],
        help="""
        BeautifulSoup: Basic HTML parsing, faster but less sophisticated
        ScrapeGraph: AI-powered scraping, better at understanding content but slower
        Crawl4ai: Advanced async crawler with good JavaScript support
        """
    )
    
    if st.button("Process URL", type="primary"):
        if url:
            # Check if URL has changed
            if url != st.session_state.current_url:
                reset_chat_history()
                st.session_state.current_url = url
            
            with st.spinner("Processing URL... This may take a moment."):
                try:
                    st.session_state.rag.crawl_and_process(url, scraping_method)
                    st.session_state.url_processed = True
                    st.success("URL processed successfully!")
                    st.rerun()  # Rerun the app to refresh the chat interface
                except Exception as e:
                    st.error(f"Error processing URL: {str(e)}")
        else:
            st.warning("Please enter a URL")
    
    st.divider()
    st.markdown("### How to use")
    st.markdown("""
    1. Enter a webpage URL in the input field
    2. Click 'Process URL' to analyze the content
    3. Ask questions about the webpage content
    4. Get AI-powered answers based on the content
    """)

# Main chat interface
st.divider()

# Display chat messages
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"""
        <div class="user-message">
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="assistant-message">
            {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Chat input
if st.session_state.url_processed:
    question = st.chat_input("Ask a question about the webpage...")
    if question:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Get answer from RAG
        with st.spinner("Thinking..."):
            try:
                answer = st.session_state.rag.ask_question(
                    question,
                    [(msg["content"], msg["content"]) for msg in st.session_state.chat_history if msg["role"] == "assistant"]
                )
                # Add assistant message to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")
else:
    st.info("👈 Please process a URL first using the sidebar")

# Footer
st.divider()
st.markdown("Built with Streamlit, LangChain, and Groq") 