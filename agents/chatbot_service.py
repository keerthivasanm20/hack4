import os
import asyncio
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import requests
from bs4 import BeautifulSoup
import boto3
# LangChain imports - updated for AWS Bedrock with ChromaDB
from langchain_aws import ChatBedrock, BedrockEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType, create_react_agent
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.tools import Tool, BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain_core.tools import tool

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

class MCPServerTool(BaseTool):
    """MCP (Model Context Protocol) Server Tool for specialized data engineering queries"""
    
    name: str = "mcp_server"
    description: str = """
    Use this tool for specialized data engineering queries that require deep technical knowledge.
    This tool can help with:
    - Complex data pipeline architectures
    - Advanced database optimization strategies
    - Cloud platform specific implementations
    - Best practices for specific technologies
    - Code examples and implementation details
    """
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Execute MCP server query"""
        try:
            # Simulate MCP server response with specialized data engineering knowledge
            query_lower = query.lower()
            
            if any(keyword in query_lower for keyword in ['architecture', 'design pattern', 'system design']):
                return self._get_architecture_guidance(query)
            elif any(keyword in query_lower for keyword in ['optimization', 'performance', 'tuning']):
                return self._get_optimization_guidance(query)
            elif any(keyword in query_lower for keyword in ['cloud', 'aws', 'azure', 'gcp']):
                return self._get_cloud_guidance(query)
            elif any(keyword in query_lower for keyword in ['code', 'implementation', 'example']):
                return self._get_code_examples(query)
            else:
                return self._get_general_guidance(query)
                
        except Exception as e:
            return f"MCP Server error: {str(e)}"
    
    def _get_architecture_guidance(self, query: str) -> str:
        return """
        **Data Architecture Best Practices:**
        
        1. **Lambda Architecture**: Combine batch and stream processing
           - Batch Layer: Historical data processing (Spark, Hadoop)
           - Speed Layer: Real-time processing (Kafka, Storm)
           - Serving Layer: Query interface (Druid, Cassandra)
        
        2. **Kappa Architecture**: Stream-first approach
           - Single processing engine for both real-time and batch
           - Use Kafka + Spark Streaming or Flink
        
        3. **Data Mesh**: Decentralized data architecture
           - Domain-oriented ownership
           - Self-serve data infrastructure
           - Product thinking for data
        
        4. **Modern Data Stack**:
           - Ingestion: Fivetran, Airbyte, Singer
           - Storage: Snowflake, BigQuery, Databricks
           - Transformation: dbt, Dataform
           - Orchestration: Airflow, Prefect, Dagster
        """
    
    def _get_optimization_guidance(self, query: str) -> str:
        query_lower = query.lower()
        
        if any(keyword in query_lower for keyword in ['sql', 'query', 'select', 'where', 'join', 'index']):
            return self._get_sql_optimization_guidance(query)
        elif any(keyword in query_lower for keyword in ['spark', 'dataframe', 'rdd']):
            return self._get_spark_optimization_guidance()
        else:
            return """
            **Performance Optimization Strategies:**
            
            1. **Database Optimization**:
               - Indexing: B-tree, Hash, Bitmap indexes
               - Partitioning: Range, Hash, List partitioning
               - Query optimization: EXPLAIN plans, statistics
            
            2. **Spark Optimization**:
               - Memory tuning: executor memory, driver memory
               - Parallelism: optimal partition count
               - Caching: persist frequently used DataFrames
               - File formats: Parquet over CSV
            
            3. **Data Pipeline Optimization**:
               - Incremental processing vs full refresh
               - Data compression (Snappy, LZ4, Gzip)
               - Column pruning and predicate pushdown
               - Resource scheduling and auto-scaling
            """
    
    def _get_sql_optimization_guidance(self, query: str) -> str:
        return f"""
        **SQL Query Optimization for: "{query}"**
        
        **Immediate Optimizations:**
        
        1. **Avoid SELECT *** - Specify only needed columns:
        ```sql
        -- Instead of: SELECT * FROM users WHERE created_date > '2023-01-01' ORDER BY id LIMIT 1000
        SELECT id, username, email, created_date 
        FROM users 
        WHERE created_date > '2023-01-01' 
        ORDER BY id 
        LIMIT 1000;
        ```
        
        2. **Add Proper Indexing:**
        ```sql
        -- Create composite index for WHERE and ORDER BY
        CREATE INDEX idx_users_created_date_id ON users(created_date, id);
        
        -- Or separate indexes if needed
        CREATE INDEX idx_users_created_date ON users(created_date);
        CREATE INDEX idx_users_id ON users(id);
        ```
        
        3. **Optimize the WHERE clause:**
        ```sql
        -- Use proper date comparison
        SELECT id, username, email, created_date 
        FROM users 
        WHERE created_date >= DATE('2023-01-01')
        AND created_date < DATE('2024-01-01')  -- More efficient than open-ended range
        ORDER BY id 
        LIMIT 1000;
        ```
        
        4. **Consider Pagination for Large Results:**
        ```sql
        -- Use cursor-based pagination instead of OFFSET
        SELECT id, username, email, created_date 
        FROM users 
        WHERE created_date >= '2023-01-01'
        AND id > 12345  -- Last seen ID from previous page
        ORDER BY id 
        LIMIT 1000;
        ```
        
        5. **Query Analysis Tools:**
        ```sql
        -- PostgreSQL
        EXPLAIN (ANALYZE, BUFFERS) 
        SELECT id, username, email, created_date 
        FROM users 
        WHERE created_date > '2023-01-01' 
        ORDER BY id 
        LIMIT 1000;
        
        -- MySQL
        EXPLAIN FORMAT=JSON 
        SELECT id, username, email, created_date 
        FROM users 
        WHERE created_date > '2023-01-01' 
        ORDER BY id 
        LIMIT 1000;
        ```
        
        **Advanced Optimizations:**
        - **Partitioning**: Partition by created_date for time-based queries
        - **Materialized Views**: For frequently accessed date ranges
        - **Query Hints**: Database-specific optimization hints
        - **Connection Pooling**: Reduce connection overhead
        """
    
    def _get_spark_optimization_guidance(self) -> str:
        return """
        **Apache Spark Optimization Strategies:**
        
        1. **DataFrame Optimizations:**
        ```python
        # Enable Adaptive Query Execution
        spark.conf.set("spark.sql.adaptive.enabled", "true")
        spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        
        # Optimize joins
        df.repartition("join_key").join(other_df.repartition("join_key"), "join_key")
        
        # Use broadcast for small tables
        from pyspark.sql.functions import broadcast
        large_df.join(broadcast(small_df), "key")
        ```
        
        2. **Memory and Resource Tuning:**
        ```python
        # Optimal executor configuration
        spark = SparkSession.builder \
            .config("spark.executor.memory", "4g") \
            .config("spark.executor.cores", "3") \
            .config("spark.sql.adaptive.coalescePartitions.minPartitionNum", "1") \
            .getOrCreate()
        ```
        
        3. **Caching Strategies:**
        ```python
        # Cache frequently used DataFrames
        df.cache()  # or df.persist(StorageLevel.MEMORY_AND_DISK)
        
        # Check if cached
        df.is_cached
        
        # Unpersist when done
        df.unpersist()
        ```
        """
    
    def _get_cloud_guidance(self, query: str) -> str:
        return """
        **Cloud Data Platform Guidance:**
        
        **AWS**:
        - Data Lake: S3 + Glue + Athena
        - Data Warehouse: Redshift
        - Stream Processing: Kinesis + Lambda
        - ETL: Glue, EMR, Step Functions
        
        **Azure**:
        - Data Lake: Azure Data Lake Storage Gen2
        - Data Warehouse: Synapse Analytics
        - Stream Processing: Event Hubs + Stream Analytics
        - ETL: Data Factory, Databricks
        
        **GCP**:
        - Data Lake: Cloud Storage + BigQuery
        - Stream Processing: Pub/Sub + Dataflow
        - ETL: Cloud Composer (Airflow), Dataproc
        
        **Best Practices**:
        - Use managed services for reduced operational overhead
        - Implement proper IAM and security policies
        - Cost optimization with reserved instances and spot pricing
        """
    
    def _get_code_examples(self, query: str) -> str:
        return """
        **Code Implementation Examples:**
        
        ```python
        # Spark DataFrame optimization
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import *
        
        # Optimized Spark session
        spark = SparkSession.builder \\
            .appName("OptimizedDataProcessing") \\
            .config("spark.sql.adaptive.enabled", "true") \\
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \\
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \\
            .getOrCreate()
        
        # Efficient data processing
        df = spark.read.parquet("s3://bucket/data/") \\
            .filter(col("date") >= "2023-01-01") \\
            .select("id", "value", "timestamp") \\
            .repartition(200, "id") \\
            .cache()
        ```
        
        ```sql
        -- Optimized SQL query
        WITH ranked_data AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp DESC) as rn
            FROM user_events
            WHERE event_date >= CURRENT_DATE - INTERVAL '30 days'
        )
        SELECT user_id, event_type, timestamp
        FROM ranked_data
        WHERE rn = 1;
        ```
        """
    
    def _get_general_guidance(self, query: str) -> str:
        return f"""
        **Data Engineering Guidance for: "{query}"**
        
        **General Recommendations:**
        
        1. **Data Quality**: Implement validation rules and monitoring
        2. **Scalability**: Design for horizontal scaling
        3. **Monitoring**: Use tools like DataDog, Prometheus
        4. **Documentation**: Maintain data lineage and schemas
        5. **Testing**: Unit tests for transformations, integration tests for pipelines
        
        **Tools and Technologies:**
        - **Orchestration**: Apache Airflow, Prefect, Dagster
        - **Processing**: Apache Spark, Flink, Kafka
        - **Storage**: Parquet, Delta Lake, Iceberg
        - **Quality**: Great Expectations, Monte Carlo, Datafold
        
        For more specific guidance, please provide additional context about your use case.
        """

class RAGTool(BaseTool):
    """RAG (Retrieval Augmented Generation) Tool for knowledge base queries"""
    
    name: str = "knowledge_base_search"
    description: str = """
    Use this tool to search the internal knowledge base for documented procedures, 
    best practices, and organizational knowledge about data engineering.
    """
    vector_store: Optional[Chroma] = None
    
    def __init__(self, vector_store: Optional[Chroma] = None, **kwargs):
        super().__init__(**kwargs)
        self.vector_store = vector_store
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Search knowledge base for relevant information"""
        try:
            if not self.vector_store:
                return "Knowledge base not available. Using general data engineering knowledge."
            
            # Enhanced query processing for better relevance
            enhanced_query = self._enhance_query(query)
            
            # Search for relevant documents with error handling
            try:
                docs = self.vector_store.similarity_search(enhanced_query, k=5)
            except Exception as search_error:
                print(f"Vector search error: {search_error}")
                # Try with original query
                try:
                    docs = self.vector_store.similarity_search(query, k=3)
                except Exception as fallback_error:
                    print(f"Fallback search error: {fallback_error}")
                    return f"Knowledge base search temporarily unavailable. Error: {str(fallback_error)}"
            
            if not docs:
                return "No relevant information found in knowledge base."
            
            # Format the results with better organization
            results = []
            for i, doc in enumerate(docs[:5], 1):  # Limit to 5 results max
                try:
                    # Extract source info if available
                    source = doc.metadata.get('source', f'Document {i}') if hasattr(doc, 'metadata') and doc.metadata else f'Document {i}'
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    
                    # Truncate very long content
                    if len(content) > 500:
                        content = content[:500] + "... [truncated]"
                    
                    results.append(f"**Source {i} ({source}):**\n{content}\n")
                except Exception as format_error:
                    print(f"Error formatting result {i}: {format_error}")
                    continue
            
            if not results:
                return "Error formatting search results from knowledge base."
            
            return f"**Knowledge Base Results:**\n\n" + "\n".join(results)
            
        except Exception as e:
            print(f"Knowledge base search error: {e}")
            return f"Knowledge base search error: {str(e)}"
    
    def _enhance_query(self, query: str) -> str:
        """Enhance query for better vector search results"""
        query_lower = query.lower()
        
        # Add relevant keywords for SQL queries
        if any(keyword in query_lower for keyword in ['sql', 'query', 'select', 'where', 'join']):
            return f"{query} SQL database optimization performance indexing"
        
        # Add relevant keywords for data pipeline queries
        elif any(keyword in query_lower for keyword in ['pipeline', 'etl', 'data flow']):
            return f"{query} data pipeline architecture ETL processing"
        
        # Add relevant keywords for Spark queries
        elif any(keyword in query_lower for keyword in ['spark', 'dataframe', 'big data']):
            return f"{query} Apache Spark optimization performance tuning"
        
        return query

class WebSearchTool(BaseTool):
    """Enhanced Web Search Tool for current information"""
    
    name: str = "web_search"
    description: str = """
    Use this tool to search the web for current information about data engineering technologies,
    recent updates, new tools, or trending practices. Best for queries about latest versions,
    recent releases, or current market trends.
    """
    search_tool: Any = None
    
    def __init__(self, search_tool=None, **kwargs):
        super().__init__(**kwargs)
        self.search_tool = search_tool or SimpleWebSearch()
    
    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        """Search the web for current information"""
        try:
            # Enhance query for better data engineering results
            enhanced_query = f"data engineering {query} 2024 best practices"
            results = self.search_tool.run(enhanced_query)
            
            return f"**Web Search Results:**\n{results}"
            
        except Exception as e:
            return f"Web search error: {str(e)}"

class DataEngineerChatBot:
    def __init__(self, chatbot_id: int):
        self.chatbot = ChatBot.objects.get(id=chatbot_id)
        
        # Initialize AWS Bedrock LLM
        aws_access_key_id = getattr(settings, 'AWS_ACCESS_KEY_ID', None)
        aws_secret_access_key = getattr(settings, 'AWS_SECRET_ACCESS_KEY', None)
        aws_region = getattr(settings, 'AWS_REGION', 'us-east-1')
        model_id = getattr(settings, 'BEDROCK_MODEL_ID', '')

        if not aws_access_key_id or not aws_secret_access_key:
            # Use a mock response for development
            self.llm = None
            self.embeddings = None
            self.bedrock_client = None
        else:
            try:
                # Initialize boto3 session
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=aws_region
                )
                
                # Initialize Bedrock client
                self.bedrock_client = session.client('bedrock-runtime', region_name=aws_region)
                
                # Initialize LangChain Bedrock LLM
                self.llm = ChatBedrock(
                    client=self.bedrock_client,
                    model_id=model_id,
                    model_kwargs={
                        "max_tokens": 4096,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "stop_sequences": ["\n\nHuman:"]
                    }
                )
                
                # Initialize Bedrock embeddings (using Amazon Titan)
                self.embeddings = BedrockEmbeddings(
                    client=self.bedrock_client,
                    model_id="amazon.titan-embed-text-v1"
                )
                
            except Exception as e:
                print(f"Error initializing Bedrock: {e}")
                self.llm = None
                self.embeddings = None
                self.bedrock_client = None
        
        # If Bedrock not available, use Hugging Face embeddings as fallback
        if not self.embeddings:
            try:
                print("Using Hugging Face embeddings as fallback...")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
            except Exception as e:
                print(f"Error initializing Hugging Face embeddings: {e}")
                self.embeddings = None
        
        # Initialize search tool
        try:
            self.search_tool = DuckDuckGoSearchRun()
        except Exception:
            self.search_tool = SimpleWebSearch()
        
        # Initialize knowledge base
        self.vector_store = self._load_knowledge_base()
        
        # Initialize the three tools
        self.tools = self._initialize_tools()
        
        # Initialize agent if LLM is available
        self.agent = self._initialize_agent() if self.llm else None
        
        # System prompt for data engineering assistant
        self.system_prompt = """
        You are a friendly Data Engineer Assistant who helps with technical challenges in simple, conversational language.
        
        Communication Style:
        - Use simple, everyday English (avoid jargon when possible)
        - Be conversational and helpful, like talking to a colleague
        - Give direct, practical answers
        - Provide only the most relevant information
        - Use examples and code only when they make things clearer

        Key Areas of Expertise:
        - Data pipeline design and optimization
        - ETL/ELT processes and best practices
        - Database design and query optimization
        - Data warehouse architecture
        - Big data technologies (Spark, Kafka, Airflow, etc.)
        - Cloud data platforms (AWS, GCP, Azure)
        - Data quality and performance tuning

        Guidelines:
        1. Answer the specific question asked
        2. Keep responses focused and not overwhelming
        3. Use one tool at a time for the most relevant information
        4. Explain things step by step when needed
        5. Ask follow-up questions if you need clarification
        6. Be encouraging and supportive

        Remember: Your goal is to help engineers solve problems quickly and clearly, not to impress them with technical complexity.
        """

    def _initialize_tools(self) -> List[BaseTool]:
        """Initialize the three main tools: MCP Server, RAG, and Web Search"""
        tools = [
            MCPServerTool(),
            RAGTool(vector_store=self.vector_store),
            WebSearchTool(search_tool=self.search_tool)
        ]
        return tools

    def _initialize_agent(self):
        """Initialize LangChain agent with the three tools"""
        try:
            if not self.llm:
                return None
            
            # Create agent prompt
            agent_prompt = PromptTemplate.from_template("""
            You are a friendly Data Engineer Assistant who communicates in simple, clear English.

            Available Tools:
            1. **Knowledge Base (knowledge_base_search)**: For documented procedures and organizational knowledge  
            2. **MCP Server (mcp_server)**: For technical guidance and implementation details
            3. **Web Search (web_search)**: For current information and latest trends

            **Tool Selection Strategy:**
            - Choose the MOST RELEVANT single tool for each question
            - For SQL/Database questions → Use MCP Server for optimization guidance
            - For Architecture questions → Use MCP Server for design patterns
            - For Current trends/latest info → Use Web Search
            - For documented procedures → Use Knowledge Base
            
            **Response Guidelines:**
            1. Use simple, conversational language (like talking to a colleague)
            2. Give direct answers to the specific question asked
            3. Focus on practical, actionable advice
            4. Include code examples only when they help explain the solution
            5. Keep responses focused - don't overwhelm with too much information
            6. Ask follow-up questions if you need clarification

            **Process:**
            1. Understand what the user is really asking
            2. Pick the ONE most relevant tool 
            3. Use that tool to get specific information
            4. Give a clear, helpful answer in simple English

            Tools available: {tools}
            Tool names: {tool_names}

            Question: {input}

            Thought: I need to understand the question and pick the best tool to help answer it clearly and simply.

            {agent_scratchpad}
            """)
            
            # Create ReAct agent
            agent = create_react_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=agent_prompt
            )
            
            return agent
            
        except Exception as e:
            print(f"Error initializing agent: {e}")
            return None

    def _load_knowledge_base(self) -> Optional[Chroma]:
        """Load the ChromaDB vector store from knowledge base documents."""
        try:
            if not self.embeddings:
                return None
                
            # Define knowledge base path
            knowledge_base_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 
                'knowledge_base'
            )
            
            if os.path.exists(knowledge_base_path):
                # Load documents from the knowledge base directory
                loader = DirectoryLoader(
                    knowledge_base_path,
                    glob="**/*.md",
                    loader_cls=TextLoader,
                    recursive=True
                )
                documents = loader.load()
                
                if documents:
                    # Split documents into chunks
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                    )
                    splits = text_splitter.split_documents(documents)
                    
                    # Create ChromaDB vector store
                    persist_directory = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)), 
                        'chroma_db'
                    )
                    
                    vector_store = Chroma.from_documents(
                        documents=splits,
                        embedding=self.embeddings,
                        persist_directory=persist_directory,
                        collection_name="data_engineering_knowledge"
                    )
                    
                    print(f"Loaded {len(splits)} document chunks into ChromaDB")
                    return vector_store
                else:
                    print("No documents found in knowledge base")
            else:
                print(f"Knowledge base path does not exist: {knowledge_base_path}")
                
            return None
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return None

    def _call_bedrock_directly(self, messages: List[str]) -> str:
        """Call Bedrock directly using boto3 client."""
        try:
            if not self.bedrock_client:
                return self._generate_mock_response("")
            
            # Format messages for Claude
            formatted_messages = []
            for i, msg in enumerate(messages):
                if i == 0:  # System message
                    continue
                elif i % 2 == 1:  # Human messages
                    formatted_messages.append(f"Human: {msg}")
                else:  # Assistant messages
                    formatted_messages.append(f"Assistant: {msg}")
            
            # Add the current human message
            conversation = "\n\n".join(formatted_messages)
            conversation += "\n\nAssistant:"
            
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": 0.2,
                "top_p": 0.9,
                "messages": [
                    {
                        "role": "user",
                        "content": conversation
                    }
                ]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=getattr(settings, 'BEDROCK_MODEL_ID', ''),
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text']
            
        except Exception as e:
            print(f"Error calling Bedrock directly: {e}")
            return self._generate_mock_response("")

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
            
            # Get messages and convert to LangChain format
            conversation_messages = list(conversation.messages.all().order_by('created_at'))
            
            # Take only the last 10 messages to avoid context overflow
            recent_messages = conversation_messages[-10:] if len(conversation_messages) > 10 else conversation_messages
            
            for msg in recent_messages:
                if msg.message_type == 'user':
                    messages.append(HumanMessage(content=msg.content))
                elif msg.message_type == 'assistant':
                    messages.append(AIMessage(content=msg.content))
            
            return messages
        except Conversation.DoesNotExist:
            return []
        except Exception as e:
            print(f"Error getting conversation history: {e}")
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

    def get_response_sync(self, user_message: str, conversation_id: int, user_id: int) -> Dict[str, Any]:
        """Synchronous version of get_response for Django views."""
        try:
            # Get conversation history
            chat_history = self._get_conversation_history(conversation_id)
            
            # Format conversation context
            context_messages = []
            for msg in chat_history[-5:]:  # Last 5 messages for context
                if isinstance(msg, HumanMessage):
                    context_messages.append(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_messages.append(f"Assistant: {msg.content}")
            
            conversation_context = "\n".join(context_messages) if context_messages else "This is the start of the conversation."
            
            # Enhanced user message with context
            enhanced_message = f"""
            Conversation Context:
            {conversation_context}
            
            Current Question: {user_message}
            
            Please use your available tools (MCP Server, Knowledge Base, Web Search) to provide a comprehensive answer.
            """
            
            # Generate response using sync tool usage
            assistant_response = self._fallback_tool_usage_sync(user_message)
            
            # Save messages to database
            conversation = Conversation.objects.get(id=conversation_id)
            
            # Save user message
            Message.objects.create(
                conversation=conversation,
                content=user_message,
                message_type='user'
            )
            
            # Save assistant response with metadata
            model_used = 'claude-3.7-sonnet' if (self.llm or self.bedrock_client) else 'mock'
            Message.objects.create(
                conversation=conversation,
                content=assistant_response,
                message_type='assistant',
                metadata={
                    'tools_used': ['mcp_server', 'knowledge_base_search', 'web_search'],
                    'model_used': model_used,
                    'provider': 'aws_bedrock' if (self.llm or self.bedrock_client) else 'mock',
                    'agent_mode': bool(self.agent)
                }
            )
            
            return {
                'success': True,
                'response': assistant_response,
                'metadata': {
                    'conversation_id': conversation_id,
                    'model_used': model_used,
                    'provider': 'aws_bedrock' if (self.llm or self.bedrock_client) else 'mock',
                    'tools_available': ['mcp_server', 'knowledge_base_search', 'web_search'],
                    'agent_mode': bool(self.agent)
                }
            }
            
        except Exception as e:
            print(f"Error in get_response_sync: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "I'm sorry, I encountered an error while processing your request. Please try again."
            }

    async def get_response(self, user_message: str, conversation_id: int, user_id: int) -> Dict[str, Any]:
        """Generate response using Claude 3.7 Sonnet with MCP Server, RAG, and Web Search tools."""
        try:
            # Get conversation history
            chat_history = self._get_conversation_history(conversation_id)
            
            # Format conversation context
            context_messages = []
            for msg in chat_history[-5:]:  # Last 5 messages for context
                if isinstance(msg, HumanMessage):
                    context_messages.append(f"Human: {msg.content}")
                elif isinstance(msg, AIMessage):
                    context_messages.append(f"Assistant: {msg.content}")
            
            conversation_context = "\n".join(context_messages) if context_messages else "This is the start of the conversation."
            
            # Enhanced user message with context
            enhanced_message = f"""
            Conversation Context:
            {conversation_context}
            
            Current Question: {user_message}
            
            Please use your available tools (MCP Server, Knowledge Base, Web Search) to provide a comprehensive answer.
            """
            
            # Generate response using agent or fallback methods
            if self.agent and self.llm:
                # Use LangChain agent with three tools
                try:
                    from langchain.agents import AgentExecutor
                    
                    agent_executor = AgentExecutor(
                        agent=self.agent,
                        tools=self.tools,
                        verbose=True,
                        max_iterations=3,
                        handle_parsing_errors=True
                    )
                    
                    result = await agent_executor.ainvoke({
                        "input": enhanced_message,
                        "tools": [tool.name for tool in self.tools],
                        "tool_names": ", ".join([tool.name for tool in self.tools])
                    })
                    
                    assistant_response = result.get("output", "I apologize, but I couldn't generate a proper response.")
                    
                except Exception as e:
                    print(f"Agent execution error: {e}")
                    # Fallback to direct tool usage
                    assistant_response = await self._fallback_tool_usage(user_message)
                    
            elif self.bedrock_client:
                # Use direct Bedrock call with manual tool coordination
                assistant_response = await self._direct_bedrock_with_tools(user_message)
            else:
                # Use sync tool usage for compatibility
                assistant_response = self._fallback_tool_usage_sync(user_message)
            
            # Save messages to database
            conversation = Conversation.objects.get(id=conversation_id)
            
            # Save user message
            Message.objects.create(
                conversation=conversation,
                content=user_message,
                message_type='user'
            )
            
            # Save assistant response with metadata
            model_used = 'claude-3.7-sonnet' if (self.llm or self.bedrock_client) else 'mock'
            Message.objects.create(
                conversation=conversation,
                content=assistant_response,
                message_type='assistant',
                metadata={
                    'tools_used': ['mcp_server', 'knowledge_base_search', 'web_search'] if self.agent else [],
                    'model_used': model_used,
                    'provider': 'aws_bedrock' if (self.llm or self.bedrock_client) else 'mock',
                    'agent_mode': bool(self.agent)
                }
            )
            
            return {
                'success': True,
                'response': assistant_response,
                'metadata': {
                    'conversation_id': conversation_id,
                    'model_used': model_used,
                    'provider': 'aws_bedrock' if (self.llm or self.bedrock_client) else 'mock',
                    'tools_available': ['mcp_server', 'knowledge_base_search', 'web_search'],
                    'agent_mode': bool(self.agent)
                }
            }
            
        except Exception as e:
            print(f"Error in get_response: {e}")
            return {
                'success': False,
                'error': str(e),
                'response': "I'm sorry, I encountered an error while processing your request. Please try again."
            }

    def _fallback_tool_usage_sync(self, user_message: str) -> str:
        """Synchronous fallback method to use tools selectively and provide natural responses"""
        try:
            # Check for basic greetings first - no need for tool usage
            if self._is_basic_greeting(user_message):
                return self._handle_basic_greeting(user_message)
            
            # Check for simple queries that don't need full tool usage
            if self._is_simple_query(user_message):
                return self._handle_simple_query(user_message)
            
            # Find the most relevant tool for the query
            best_tool_result = self._get_best_matching_result(user_message)
            
            # Generate a natural, conversational response
            return self._generate_natural_response(user_message, best_tool_result)
                
        except Exception as e:
            print(f"Fallback tool usage sync error: {e}")
            return self._generate_mock_response(user_message)

    def _is_basic_greeting(self, message: str) -> bool:
        """Check if message is a basic greeting"""
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening', 
                    'greetings', 'howdy', 'hiya', 'sup', 'what\'s up', 'how are you']
        message_lower = message.lower().strip()
        return any(greeting in message_lower for greeting in greetings) and len(message.split()) <= 3

    def _handle_basic_greeting(self, message: str) -> str:
        """Handle basic greetings with simple responses"""
        return """Hello! 👋 I'm your Data Engineering Assistant.

I can help you with:
• SQL query optimization
• Data pipeline design
• ETL/ELT processes
• Big data technologies (Spark, Kafka, Airflow)
• Cloud data platforms
• Database performance tuning

What data engineering challenge can I help you solve today?"""

    def _is_simple_query(self, message: str) -> bool:
        """Check if message is a simple query that doesn't need full tool usage"""
        simple_patterns = ['what is', 'what are', 'define', 'explain', 'help', 'thanks', 'thank you']
        message_lower = message.lower().strip()
        return any(pattern in message_lower for pattern in simple_patterns) and len(message.split()) <= 5

    def _handle_simple_query(self, message: str) -> str:
        """Handle simple queries with direct responses"""
        message_lower = message.lower()
        
        if 'thank' in message_lower:
            return "You're welcome! Feel free to ask me any data engineering questions."
        
        if 'help' in message_lower:
            return """I'm here to help with data engineering challenges! You can ask me about:

**Database & SQL:**
• Query optimization and performance tuning
• Indexing strategies and database design
• SQL best practices

**Data Pipelines:**
• ETL/ELT process design
• Data pipeline architecture
• Workflow orchestration

**Big Data:**
• Apache Spark optimization
• Kafka streaming
• Distributed processing

**Cloud Platforms:**
• AWS, GCP, Azure data services
• Cloud architecture patterns

Just ask your question and I'll provide detailed guidance!"""
        
        return self._generate_direct_response(message)

    def _needs_knowledge_base_search(self, message: str) -> bool:
        """Determine if message needs knowledge base search"""
        technical_keywords = ['sql', 'query', 'database', 'pipeline', 'etl', 'spark', 'kafka', 
                             'optimization', 'performance', 'architecture', 'design', 'best practices']
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in technical_keywords) and len(message.split()) > 3

    def _needs_technical_guidance(self, message: str) -> bool:
        """Determine if message needs MCP server technical guidance"""
        message_lower = message.lower()
        return any(keyword in message_lower for keyword in ['how to', 'implement', 'optimize', 'design', 
                                                           'architecture', 'pattern', 'solution', 'approach'])

    def _get_best_matching_result(self, user_message: str) -> str:
        """Find the most relevant tool result for the query"""
        message_lower = user_message.lower()
        
        # Determine which single tool is most relevant
        if self._is_sql_related(message_lower):
            # For SQL queries, prioritize MCP Server for optimization guidance
            if self._needs_technical_guidance(user_message):
                mcp_tool = MCPServerTool()
                return mcp_tool._run(user_message)
            elif self.vector_store:
                rag_tool = RAGTool(vector_store=self.vector_store)
                return rag_tool._run(user_message)
        
        elif self._is_architecture_related(message_lower):
            # For architecture questions, use MCP Server first
            mcp_tool = MCPServerTool()
            return mcp_tool._run(user_message)
        
        elif self._needs_current_info(message_lower):
            # For current trends, use web search
            web_tool = WebSearchTool(self.search_tool)
            return web_tool._run(user_message)
        
        elif self.vector_store and self._needs_knowledge_base_search(user_message):
            # Default to knowledge base for documented procedures
            rag_tool = RAGTool(vector_store=self.vector_store)
            return rag_tool._run(user_message)
        
        else:
            # For general queries, use MCP Server
            mcp_tool = MCPServerTool()
            return mcp_tool._run(user_message)

    def _is_sql_related(self, message_lower: str) -> bool:
        """Check if query is SQL related"""
        sql_keywords = ['sql', 'query', 'select', 'where', 'join', 'index', 'database', 'table', 'optimize']
        return any(keyword in message_lower for keyword in sql_keywords)

    def _is_architecture_related(self, message_lower: str) -> bool:
        """Check if query is architecture related"""
        arch_keywords = ['architecture', 'design', 'pipeline', 'system', 'structure', 'pattern']
        return any(keyword in message_lower for keyword in arch_keywords)

    def _needs_current_info(self, message_lower: str) -> bool:
        """Check if query needs current information"""
        current_keywords = ['latest', 'recent', 'new', 'current', 'trend', '2024', '2025', 'update']
        return any(keyword in message_lower for keyword in current_keywords)

    def _generate_natural_response(self, user_message: str, tool_result: str) -> str:
        """Generate a natural, conversational response from tool results"""
        try:
            # Clean up tool result - remove technical formatting
            cleaned_result = self._clean_tool_output(tool_result)
            
            # Generate natural response using Bedrock if available
            if self.bedrock_client:
                prompt = f"""
                A user asked: "{user_message}"
                
                Here's relevant technical information: {cleaned_result}
                
                Please provide a clear, conversational response in simple English that:
                1. Directly answers their question
                2. Uses simple, everyday language 
                3. Provides practical steps they can follow
                4. Includes code examples only when helpful
                5. Keeps the response focused and not overwhelming
                
                Write as if you're explaining to a colleague in a friendly conversation.
                """
                
                return self._call_bedrock_directly([prompt])
            else:
                # Fallback to formatted response
                return self._format_simple_response(user_message, cleaned_result)
                
        except Exception as e:
            print(f"Error generating natural response: {e}")
            return self._generate_direct_response(user_message)

    def _clean_tool_output(self, tool_result: str) -> str:
        """Clean and simplify tool output"""
        # Remove excessive formatting and headers
        cleaned = tool_result.replace('**', '').replace('***', '')
        cleaned = cleaned.replace('Knowledge Base Results:', '')
        cleaned = cleaned.replace('Technical Guidance:', '')
        cleaned = cleaned.replace('Web Search Results:', '')
        
        # Remove multiple newlines
        import re
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        return cleaned.strip()

    def _format_simple_response(self, user_message: str, content: str) -> str:
        """Format a simple, conversational response"""
        # Extract key points from content
        lines = content.split('\n')
        key_points = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and len(line) > 20:
                key_points.append(line)
                if len(key_points) >= 3:  # Limit to 3 key points
                    break
        
        if not key_points:
            return self._generate_direct_response(user_message)
        
        # Create conversational response
        response = f"Here's what I can help you with regarding your question:\n\n"
        
        for i, point in enumerate(key_points, 1):
            response += f"{i}. {point}\n\n"
        
        response += "Would you like me to explain any of these points in more detail?"
        
        return response

    def _generate_direct_response(self, message: str) -> str:
        """Generate a direct response without tool usage"""
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in ['sql', 'query', 'database']):
            return """For SQL and database questions, I can help with:

• **Query Optimization**: Improving performance with proper indexing and query structure
• **Database Design**: Schema design, normalization, and best practices
• **Performance Tuning**: Identifying bottlenecks and optimization strategies

Could you share your specific SQL query or database challenge?"""

        elif any(keyword in message_lower for keyword in ['pipeline', 'etl', 'data flow']):
            return """For data pipeline questions, I can assist with:

• **ETL/ELT Design**: Choosing the right approach for your data flow
• **Pipeline Architecture**: Scalable and maintainable pipeline design
• **Tool Selection**: Airflow, Spark, Kafka, and other pipeline tools

What specific pipeline challenge are you working on?"""

        elif any(keyword in message_lower for keyword in ['spark', 'big data']):
            return """For Apache Spark and big data, I can help with:

• **Performance Optimization**: Memory tuning, partitioning, caching strategies
• **Code Optimization**: Efficient transformations and actions
• **Cluster Configuration**: Resource allocation and scaling

What Spark challenge can I help you solve?"""

        else:
            return """I'm your Data Engineering Assistant! I can help you with various technical challenges:

• **SQL & Databases**: Query optimization, schema design, performance tuning
• **Data Pipelines**: ETL/ELT design, architecture patterns, tool selection
• **Big Data**: Spark optimization, distributed processing, streaming
• **Cloud Platforms**: AWS, GCP, Azure data services and architecture

What specific data engineering question do you have?"""

    def _format_tool_results(self, user_message: str, results: List[str]) -> str:
        """Format tool results into a clean, organized response"""
        if not results:
            return self._generate_direct_response(user_message)
        
        formatted_response = f"## Response to: {user_message}\n\n"
        
        for result in results:
            # Clean up the formatting
            if "Knowledge Base Search" in result:
                formatted_response += "### 📚 From Knowledge Base:\n"
                content = result.split(":**\n", 1)[1] if ":**\n" in result else result
                formatted_response += content.strip() + "\n\n"
            elif "Technical Guidance" in result:
                formatted_response += "### 🔧 Technical Guidance:\n"
                content = result.split(":**\n", 1)[1] if ":**\n" in result else result
                formatted_response += content.strip() + "\n\n"
            elif "Current Information" in result:
                formatted_response += "### 🌐 Current Information:\n"
                content = result.split(":**\n", 1)[1] if ":**\n" in result else result
                formatted_response += content.strip() + "\n\n"
        
        return formatted_response.strip()

    async def _fallback_tool_usage(self, user_message: str) -> str:
        """Async fallback method to use tools selectively and provide natural responses"""
        try:
            # Check for basic greetings first - no need for tool usage
            if self._is_basic_greeting(user_message):
                return self._handle_basic_greeting(user_message)
            
            # Check for simple queries that don't need full tool usage
            if self._is_simple_query(user_message):
                return self._handle_simple_query(user_message)
            
            # Find the most relevant tool for the query
            best_tool_result = self._get_best_matching_result(user_message)
            
            # Generate a natural, conversational response
            return self._generate_natural_response(user_message, best_tool_result)
                
        except Exception as e:
            print(f"Fallback tool usage error: {e}")
            return self._generate_mock_response(user_message)

    async def _direct_bedrock_with_tools(self, user_message: str) -> str:
        """Use Bedrock directly with coordinated tool usage"""
        try:
            # Determine which tools to use based on query analysis
            analysis_prompt = f"""
            Analyze this data engineering question and determine which tools would be most helpful:
            Question: "{user_message}"
            
            Available tools:
            1. MCP Server - for technical architecture and implementation guidance
            2. Knowledge Base - for documented procedures and best practices  
            3. Web Search - for current trends and latest information
            
            Respond with: tool1,tool2,tool3 (comma separated list of recommended tools)
            """
            
            tool_selection = self._call_bedrock_directly([analysis_prompt])
            
            # Use selected tools
            tool_results = []
            
            if 'mcp_server' in tool_selection.lower():
                mcp_tool = MCPServerTool()
                result = mcp_tool._run(user_message)
                tool_results.append(f"**Technical Guidance:**\n{result}")
            
            if 'knowledge_base' in tool_selection.lower() and self.vector_store:
                rag_tool = RAGTool(vector_store=self.vector_store)
                result = rag_tool._run(user_message)
                tool_results.append(f"**Knowledge Base:**\n{result}")
            
            if 'web_search' in tool_selection.lower():
                web_tool = WebSearchTool(self.search_tool)
                result = web_tool._run(user_message)
                tool_results.append(f"**Current Information:**\n{result}")
            
            # Generate final response
            combined_results = "\n\n".join(tool_results)
            final_prompt = f"""
            Based on the following information, provide a comprehensive answer to: "{user_message}"
            
            {combined_results}
            
            Synthesize this information into a clear, actionable response with code examples where appropriate.
            """
            
            return self._call_bedrock_directly([final_prompt])
            
        except Exception as e:
            print(f"Direct Bedrock with tools error: {e}")
            return self._generate_mock_response(user_message)

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
            content=f"""Hello! 👋 I'm your Data Engineering Assistant.

I can help you with:
• SQL query optimization and database design
• Data pipeline architecture and ETL/ELT processes  
• Big data technologies (Spark, Kafka, Airflow)
• Cloud platforms (AWS, GCP, Azure)
• Performance tuning and troubleshooting

What data engineering challenge can I help you solve today?""",
            message_type='assistant'
        )
        
        return conversation

    def delete_conversation(self, conversation_id: int, user_id: int) -> Dict[str, Any]:
        """Delete a specific conversation and all its messages."""
        try:
            conversation = Conversation.objects.get(id=conversation_id, user_id=user_id)
            conversation_title = conversation.title
            
            # Delete all messages in the conversation (cascade should handle this, but being explicit)
            Message.objects.filter(conversation=conversation).delete()
            
            # Delete the conversation
            conversation.delete()
            
            return {
                'success': True,
                'message': f"Conversation '{conversation_title}' has been deleted successfully.",
                'conversation_id': conversation_id
            }
        except Conversation.DoesNotExist:
            return {
                'success': False,
                'error': 'Conversation not found or you do not have permission to delete it.'
            }
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return {
                'success': False,
                'error': f"Failed to delete conversation: {str(e)}"
            }

    def delete_all_conversations(self, user_id: int) -> Dict[str, Any]:
        """Delete all conversations for a specific user."""
        try:
            from django.contrib.auth.models import User
            user = User.objects.get(id=user_id)
            
            # Get count before deletion
            conversation_count = Conversation.objects.filter(user=user, chatbot=self.chatbot).count()
            
            # Delete all messages for user's conversations with this chatbot
            Message.objects.filter(
                conversation__user=user,
                conversation__chatbot=self.chatbot
            ).delete()
            
            # Delete all conversations for this user and chatbot
            Conversation.objects.filter(user=user, chatbot=self.chatbot).delete()
            
            return {
                'success': True,
                'message': f"Successfully deleted {conversation_count} conversations and all their messages.",
                'deleted_count': conversation_count
            }
        except User.DoesNotExist:
            return {
                'success': False,
                'error': 'User not found.'
            }
        except Exception as e:
            print(f"Error deleting all conversations: {e}")
            return {
                'success': False,
                'error': f"Failed to delete conversations: {str(e)}"
            }

    def clear_conversation_messages(self, conversation_id: int, user_id: int) -> Dict[str, Any]:
        """Clear all messages in a conversation but keep the conversation."""
        try:
            conversation = Conversation.objects.get(id=conversation_id, user_id=user_id)
            
            # Get count before deletion
            message_count = Message.objects.filter(conversation=conversation).count()
            
            # Delete all messages in the conversation
            Message.objects.filter(conversation=conversation).delete()
            
            # Add a fresh welcome message
            Message.objects.create(
                conversation=conversation,
                content=f"""Hello! 👋 I'm your Data Engineering Assistant.

I can help you with:
• SQL query optimization and database design
• Data pipeline architecture and ETL/ELT processes  
• Big data technologies (Spark, Kafka, Airflow)
• Cloud platforms (AWS, GCP, Azure)
• Performance tuning and troubleshooting

What data engineering challenge can I help you solve today?""",
                message_type='assistant'
            )
            
            return {
                'success': True,
                'message': f"Conversation messages cleared successfully. Deleted {message_count} messages.",
                'conversation_id': conversation_id,
                'deleted_message_count': message_count
            }
        except Conversation.DoesNotExist:
            return {
                'success': False,
                'error': 'Conversation not found or you do not have permission to modify it.'
            }
        except Exception as e:
            print(f"Error clearing conversation messages: {e}")
            return {
                'success': False,
                'error': f"Failed to clear conversation messages: {str(e)}"
            }

    def get_user_conversations(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all conversations for a user with this chatbot."""
        try:
            from django.contrib.auth.models import User
            user = User.objects.get(id=user_id)
            
            conversations = Conversation.objects.filter(
                user=user, 
                chatbot=self.chatbot
            ).order_by('-updated_at')
            
            conversation_list = []
            for conv in conversations:
                # Get last message for preview
                last_message = conv.messages.last()
                message_count = conv.messages.count()
                
                conversation_list.append({
                    'id': conv.id,
                    'title': conv.title,
                    'created_at': conv.created_at,
                    'updated_at': conv.updated_at,
                    'message_count': message_count,
                    'last_message': {
                        'content': last_message.content[:100] + '...' if last_message and len(last_message.content) > 100 else last_message.content if last_message else '',
                        'message_type': last_message.message_type if last_message else '',
                        'created_at': last_message.created_at if last_message else None
                    } if last_message else None
                })
            
            return conversation_list
        except User.DoesNotExist:
            return []
        except Exception as e:
            print(f"Error getting user conversations: {e}")
            return []

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
