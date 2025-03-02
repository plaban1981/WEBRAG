import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
from scrapegraphai.graphs import SmartScraperGraph
import asyncio
from functools import partial
import sys
from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig
from langchain_community.document_loaders import TextLoader

import chromadb
from chromadb.config import Settings
import os
chroma_setting = Settings(anonymized_telemetry=False)
persist_directory = "chroma_db"
collection_metadata = {"hnsw:space": "cosine"}
client = chromadb.PersistentClient(path=persist_directory, settings=chroma_setting)
# Set Windows event loop policy
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# Apply nest_asyncio to allow nested event loops
import nest_asyncio  # Import nest_asyncio module for asynchronous operations
nest_asyncio.apply()  # Apply nest_asyncio to resolve any issues with asyncio event loop

# Load environment variables
load_dotenv()
print(os.getenv("GROQ_API_KEY"))

class WebRAG:
    def __init__(self):
        # Initialize Groq
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="mixtral-8x7b-32768"
        )
        self.response_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name="DeepSeek-R1-Distill-Llama-70B",
            temperature=0.6,
            max_tokens=2048,
        )
        # Initialize embeddings
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=r"C:\Users\PLNAYAK\Documents\New_Frontiers\claims_doc_ai\models\BAAI",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        self.vector_store =  Chroma(embedding_function= self.embeddings,
                        client = client,
                    persist_directory=persist_directory,
                    client_settings=chroma_setting,
                    )
        # self.qa_chain = None

    def crawl_webpage_bs4(self, url):
        """Crawl webpage using BeautifulSoup"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content from relevant tags
        text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])
        content = ' '.join([elem.get_text(strip=True) for elem in text_elements])
        
        # Clean up whitespace
        content = ' '.join(content.split())
        return content

    # Crawl4ai
    async def crawl_webpage_crawl4ai_async(self, url):
        """Crawl webpage using Crawl4ai asynchronously"""
        try:
            crawler_run_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=url, config=crawler_run_config)
                return result.markdown
        except Exception as e:
            raise Exception(f"Error in Crawl4ai async: {str(e)}")

    def crawl_webpage_crawl4ai(self, url):
        """Synchronous wrapper for crawl4ai"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        try:
            return loop.run_until_complete(self.crawl_webpage_crawl4ai_async(url))
        except Exception as e:
            raise Exception(f"Error in Crawl4ai: {str(e)}")

    def crawl_webpage_scrapegraph(self, url):
        """Crawl webpage using ScrapeGraphAI"""
        try:
            # First try with Groq
            graph_config = {
                "llm": {
                    "api_key": os.getenv("GROQ_API_KEY"),
                    "model": "groq/mixtral-8x7b-32768",
                },
                "verbose": True,
                "headless": True,
                "disable_async": True  # Use synchronous mode
            }
            
            scraper = SmartScraperGraph(
                prompt="Extract all the useful textual content from the webpage",
                source=url,
                config=graph_config
            )
            
            # Use synchronous run
            result = scraper.run()
            print("Groq scraping successful")
            return str(result)
            
        except Exception as e:
            print(f"Groq scraping failed, falling back to Ollama: {str(e)}")
            try:
                # Fallback to Ollama
                graph_config = {
                    "llm": {
                        "model": "ollama/deepseek-r1:8b",
                        "temperature": 0,
                        "max_tokens": 2048,
                        "format": "json",
                        "base_url": "http://localhost:11434",
                    },
                    "embeddings": {
                        "model": "ollama/nomic-embed-text",
                        "base_url": "http://localhost:11434",
                    },
                    "verbose": True,
                    "disable_async": True  # Use synchronous mode
                }
                
                scraper = SmartScraperGraph(
                    prompt="Extract all the useful textual content from the webpage",
                    source=url,
                    config=graph_config
                )
                
                result = scraper.run()
                print("Ollama scraping successful")
                return str(result)
                
            except Exception as e2:
                raise Exception(f"Both Groq and Ollama scraping failed: {str(e2)}")

    def crawl_and_process(self, url, scraping_method="beautifulsoup"):
        """Crawl the URL and process the content"""
        try:
            # Validate URL
            if not url.startswith(('http://', 'https://')):
                raise ValueError("Invalid URL. Please include http:// or https://")
            
            # Crawl the website using selected method
            if scraping_method == "beautifulsoup":
                content = self.crawl_webpage_bs4(url)
            elif scraping_method == "crawl4ai":
                content = self.crawl_webpage_crawl4ai(url)
            else:  # scrapegraph
                content = self.crawl_webpage_scrapegraph(url)
            
            if not content:
                raise ValueError("No content found at the specified URL")
            
            # Clean the content of any problematic characters
            content = content.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Create a temporary file with proper encoding
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            try:
                # Load and process the document
                docs = TextLoader(temp_path, encoding='utf-8').load()
                docs = [Document(page_content=doc.page_content, metadata={"source": url}) for doc in docs]
                chunks = self.text_splitter.split_documents(docs)
                print(f"Length of chunks: {len(chunks)}")
                print(f"First chunk: {chunks[0].metadata['source']}")
                
                # Check if path exists
                data_exists = False
                existing_urls = []
                
                if os.path.exists("chroma_db"):
                    # Check if the URL is already in the metadata
                    print(f"Checking if URL {url} is already in the metadata")
                    try:
                        self.vectorstore = Chroma(
                        embedding_function=self.embeddings,
                        client=client,
                        persist_directory=persist_directory
                        )
                        entities = self.vector_store.get(include=["metadatas"])
                        print(f"Entities: {len(entities['metadatas'])}")
                        if len(entities['metadatas']) > 0:
                            for entry in entities['metadatas']:
                                #print(f"Entry: {entry}")
                                existing_urls.append(entry["source"])
                    except Exception as e:
                        print(f"Error checking existing URLs: {str(e)}")
                print(f"Existing URLs: {set(existing_urls)}")
                if url in set(existing_urls):
                    data_exists = True
                    print(f"URL {url} already exists in the vector store")
                    # Load the existing vector store
                else:
                    # Add new documents to the vector store
                    MAX_BATCH_SIZE = 100
                    for i in range(0,len(chunks),MAX_BATCH_SIZE):
                        #print(f"start of processing: {i}")
                        i_end = min(len(chunks),i+MAX_BATCH_SIZE)
                        #print(f"end of processing: {i_end}")
                        batch = chunks[i:i_end]
                        #
                        self.vectorstore.add_documents(batch)
                        print(f"vectors for batch {i} to {i_end} stored successfully...")
                    
                
                # Create QA chain
                self.qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.response_llm,
                    retriever=self.vector_store.as_retriever(search_type="similarity",
                                                             search_kwargs={"k": 5,"filter":{"source": url}}),
                    return_source_documents=True
                )
            
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            raise Exception(f"Error processing URL: {str(e)}")

    def ask_question(self, question, chat_history=[]):
        """Ask a question about the processed content"""
        try:
            if not self.qa_chain:
                raise ValueError("Please crawl and process a URL first")
            
            response = self.qa_chain.invoke({"question": question, "chat_history": chat_history[:4000]})
            print(f"Response: {response}")
            final_answer = response["answer"].split("</think>\n\n")[-1]
            return final_answer 
        except Exception as e:
            raise Exception(f"Error generating response: {str(e)}")

def main():
    # Initialize the RAG system
    rag = WebRAG()
    
    # Get URL from user
    url = input("Enter the URL to process: ")
    print("Processing URL... This may take a moment.")
    scraping_method = input("Choose scraping method (beautifulsoup or scrapegraph or crawl4ai): ")
    rag.crawl_and_process(url, scraping_method)
    
    # Interactive Q&A loop
    chat_history = []
    while True:
        question = input("\nEnter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break
            
        answer = rag.ask_question(question, chat_history)
        print("\nAnswer:", answer)
        chat_history.append((question, answer))

if __name__ == "__main__":
    main() 