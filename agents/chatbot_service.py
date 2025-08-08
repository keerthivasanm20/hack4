import os
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import requests
from bs4 import BeautifulSoup

# LangChain imports - updated for new structure
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

# Django imports
from django.conf import settings
from .models import ChatBot, Conversation, Message

class SimpleWebSearch:
    """Simple web search implementation as fallback"""
    
    def run(self, query: str) -> str:
        try:
            # Simple DuckDuckGo search
            search_url = f"https://duckduckgo.com/html/?q={query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                
                # Extract search results
                for result in soup.find_all('div', class_='result')[:3]:
                    title_elem = result.find('h2')
                    snippet_elem = result.find('div', class_='result__snippet')
                    
                    if title_elem and snippet_elem:
                        title = title_elem.get_text().strip()
                        snippet = snippet_elem.get_text().strip()
                        results.append(f"{title}: {snippet}")
                
                return "\n".join(results) if results else "No relevant search results found."
            
            return "Search temporarily unavailable."
            
        except Exception as e:
            print(f"Web search error: {e}")
            return "Search temporarily unavailable."

class DataEngineerChatBot:
    def __init__(self, chatbot_id: int):
        self.chatbot = ChatBot.objects.get(id=chatbot_id)
        
        # Initialize OpenAI LLM
        openai_api_key = getattr(settings, 'OPENAI_API_KEY', None)
        if not openai_api_key or openai_api_key == 'your-openai-api-key-here':
            # Use a mock response for development
            self.llm = None
            self.embeddings = None
        else:
            self.llm = ChatOpenAI(
                temperature=0.2,
                model="gpt-4",
                openai_api_key=openai_api_key
            )
            self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        
        # Initialize search tool
        try:
            self.search_tool = DuckDuckGoSearchRun()
        except Exception:
            self.search_tool = SimpleWebSearch()
        
        # Initialize knowledge base
        self.vector_store = self._load_knowledge_base()
        
        # System prompt for data engineering assistant
        self.system_prompt = """
        You are an expert Data Engineer Assistant designed to help data engineers with their technical challenges.
        Your primary goal is to minimize the need for engineers to consult with architects by providing comprehensive, 
        accurate, and actionable guidance.

        Key Areas of Expertise:
        - Data pipeline design and optimization
        - ETL/ELT processes and best practices
        - Database design and query optimization
        - Data warehouse architecture
        - Big data technologies (Spark, Kafka, Airflow, etc.)
        - Cloud data platforms (AWS, GCP, Azure)
        - Data quality and validation
        - Performance tuning and troubleshooting
        - Data modeling and schema design
        - Streaming data processing
        - Data governance and compliance

        Guidelines:
        1. Always provide practical, implementable solutions
        2. Include code examples when relevant
        3. Consider scalability and performance implications
        4. Suggest best practices and industry standards
        5. When uncertain, use web search to find current information
        6. Provide multiple approaches when applicable
        7. Include relevant documentation links
        8. Consider security and compliance aspects

        Use the available knowledge base and web search when needed to provide accurate, up-to-date information.
        """

    def _load_knowledge_base(self) -> Optional[FAISS]:
        """Load the vector store from knowledge base documents."""
        try:
            if not self.embeddings:
                return None
                
            knowledge_base_path = getattr(settings, 'CHATBOT_KNOWLEDGE_BASE_PATH', None)
            if knowledge_base_path and os.path.exists(knowledge_base_path):
                # Load documents from the knowledge base directory
                loader = DirectoryLoader(
                    knowledge_base_path,
                    glob="**/*.md",
                    loader_cls=TextLoader
                )
                documents = loader.load()
                
                if documents:
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(documents)
                    
                    # Create vector store
                    vector_store = FAISS.from_documents(splits, self.embeddings)
                    return vector_store
            return None
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return None

    def _search_web(self, query: str) -> str:
        """Search the web for additional information."""
        try:
            if self.chatbot.web_search_enabled:
                return self.search_tool.run(query)
            return ""
        except Exception as e:
            print(f"Web search error: {e}")
            return ""

    def _get_conversation_history(self, conversation_id: int) -> List[BaseMessage]:
        """Retrieve conversation history as LangChain messages."""
        try:
            conversation = Conversation.objects.get(id=conversation_id)
            messages = []
            
            for msg in conversation.messages.all()[-10:]:  # Last 10 messages
                if msg.message_type == 'user':
                    messages.append(HumanMessage(content=msg.content))
                elif msg.message_type == 'assistant':
                    messages.append(AIMessage(content=msg.content))
            
            return messages
        except Conversation.DoesNotExist:
            return []

    def _generate_mock_response(self, user_message: str) -> str:
        """Generate a mock response when OpenAI is not available."""
        message_lower = user_message.lower()
        
        if any(keyword in message_lower for keyword in ['pipeline', 'etl', 'data flow']):
            return """For data pipeline design, I recommend considering these key aspects:

1. **Data Ingestion**: Use tools like Apache Kafka for real-time streaming or batch processing with tools like Apache Airflow
2. **Data Processing**: Consider Apache Spark for large-scale data processing
3. **Data Storage**: Choose between data lakes (S3, ADLS) and data warehouses (Snowflake, BigQuery) based on your use case
4. **Monitoring**: Implement comprehensive logging and monitoring with tools like DataDog or custom solutions

Would you like me to elaborate on any specific aspect of your pipeline requirements?"""

        elif any(keyword in message_lower for keyword in ['database', 'sql', 'query']):
            return """For database optimization, here are some best practices:

1. **Indexing Strategy**: Create appropriate indexes on frequently queried columns
2. **Query Optimization**: Use EXPLAIN plans to identify bottlenecks
3. **Partitioning**: Consider table partitioning for large datasets
4. **Connection Pooling**: Implement connection pooling to manage database connections efficiently

```sql
-- Example: Creating an index for better query performance
CREATE INDEX idx_user_created_date ON users(created_date);

-- Example: Analyzing query performance
EXPLAIN ANALYZE SELECT * FROM users WHERE created_date > '2023-01-01';
```

What specific database challenges are you facing?"""

        elif any(keyword in message_lower for keyword in ['spark', 'big data', 'distributed']):
            return """For Apache Spark and big data processing:

1. **Cluster Configuration**: Optimize executor memory and cores based on your data size
2. **Data Partitioning**: Properly partition your data to avoid shuffle operations
3. **Caching**: Use `.cache()` or `.persist()` for frequently accessed DataFrames
4. **File Formats**: Use Parquet or Delta Lake for better performance

```python
# Example: Optimizing Spark job
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("DataProcessing") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .getOrCreate()

# Read and process data efficiently
df = spark.read.parquet("path/to/data") \
    .repartition(200) \
    .cache()
```

What specific Spark optimization challenges are you working on?"""

        else:
            return f"""I understand you're asking about: "{user_message}"

As your Data Engineering Assistant, I'm here to help with various technical challenges including:

• Data pipeline architecture and design
• ETL/ELT process optimization
• Database performance tuning
• Big data technologies (Spark, Kafka, Airflow)
• Cloud data platforms and services
• Data quality and governance
• Real-time streaming solutions

Could you provide more specific details about your current challenge? This will help me give you more targeted guidance and practical solutions.

For example:
- What technology stack are you currently using?
- What specific problem are you trying to solve?
- Are there any performance or scalability requirements?

*Note: I can provide more detailed, real-time information when connected to live data sources and search capabilities.*"""

    async def get_response(self, user_message: str, conversation_id: int, user_id: int) -> Dict[str, Any]:
        """Generate response using RAG and web search."""
        try:
            # Get conversation history
            chat_history = self._get_conversation_history(conversation_id)
            
            # Search knowledge base if available
            relevant_docs = ""
            if self.vector_store:
                docs = self.vector_store.similarity_search(user_message, k=3)
                relevant_docs = "\n\n".join([doc.page_content for doc in docs])
            
            # Search web for additional context
            web_results = ""
            if self.chatbot.web_search_enabled and any(keyword in user_message.lower() 
                for keyword in ['latest', 'current', 'new', 'recent', 'update', '2024', '2023']):
                web_query = f"data engineering {user_message}"
                web_results = self._search_web(web_query)
            
            # Generate response
            if self.llm:
                # Use actual OpenAI model
                context = f"""
                Knowledge Base Context:
                {relevant_docs}
                
                Web Search Results:
                {web_results}
                """
                
                enhanced_message = f"{user_message}\n\nContext: {context}" if context.strip() else user_message
                
                messages = [
                    HumanMessage(content=self.system_prompt),
                    *chat_history,
                    HumanMessage(content=enhanced_message)
                ]
                
                response = await self.llm.agenerate([messages])
                assistant_response = response.generations[0][0].text
            else:
                # Use mock response for development
                assistant_response = self._generate_mock_response(user_message)
            
            # Save messages to database
            conversation = Conversation.objects.get(id=conversation_id)
            
            # Save user message
            Message.objects.create(
                conversation=conversation,
                content=user_message,
                message_type='user'
            )
            
            # Save assistant response with metadata
            Message.objects.create(
                conversation=conversation,
                content=assistant_response,
                message_type='assistant',
                metadata={
                    'has_web_search': bool(web_results),
                    'has_knowledge_base': bool(relevant_docs),
                    'sources_used': (['knowledge_base'] if relevant_docs else []) + (['web_search'] if web_results else []),
                    'model_used': 'gpt-4' if self.llm else 'mock'
                }
            )
            
            return {
                'success': True,
                'response': assistant_response,
                'metadata': {
                    'has_web_search': bool(web_results),
                    'has_knowledge_base': bool(relevant_docs),
                    'conversation_id': conversation_id,
                    'model_used': 'gpt-4' if self.llm else 'mock'
                }
            }
            
        except Exception as e:
            print(f"Error in get_response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "I'm sorry, I encountered an error while processing your request. Please try again."
            }

    def create_conversation(self, user_id: int, title: str) -> Conversation:
        """Create a new conversation."""
        from django.contrib.auth.models import User
        user = User.objects.get(id=user_id)
        
        conversation = Conversation.objects.create(
            user=user,
            chatbot=self.chatbot,
            title=title
        )
        
        # Add welcome message
        Message.objects.create(
            conversation=conversation,
            content=f"Hello! I'm your Data Engineering Assistant. I'm here to help you with data pipelines, ETL processes, database optimization, and other data engineering challenges. How can I assist you today?",
            message_type='assistant'
        )
        
        return conversation

class ChatBotManager:
    """Manager class for handling multiple chatbots."""
    
    @staticmethod
    def get_available_chatbots():
        """Get all available chatbots."""
        return ChatBot.objects.filter(is_active=True)
    
    @staticmethod
    def get_chatbot_instance(chatbot_id: int):
        """Get a chatbot instance."""
        chatbot = ChatBot.objects.get(id=chatbot_id)
        
        if chatbot.chatbot_type == 'data_engineer':
            return DataEngineerChatBot(chatbot_id)
        # Add other chatbot types here
        else:
            return DataEngineerChatBot(chatbot_id)  # Default fallback
    
    @staticmethod
    def get_user_conversations(user_id: int, chatbot_id: int = None):
        """Get user's conversations."""
        conversations = Conversation.objects.filter(user_id=user_id)
        if chatbot_id:
            conversations = conversations.filter(chatbot_id=chatbot_id)
        return conversations.order_by('-updated_at')
