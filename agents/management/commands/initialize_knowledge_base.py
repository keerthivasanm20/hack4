from django.core.management.base import BaseCommand
from django.conf import settings
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader


class Command(BaseCommand):
    help = 'Initialize ChromaDB knowledge base with data engineering documents'

    def add_arguments(self, parser):
        parser.add_argument(
            '--rebuild',
            action='store_true',
            help='Rebuild the entire knowledge base from scratch',
        )

    def handle(self, *args, **options):
        self.stdout.write('Starting ChromaDB knowledge base initialization...')
        
        # Check if knowledge base directory exists
        knowledge_base_path = os.path.join(
            settings.BASE_DIR, 
            'knowledge_base'
        )
        
        if not os.path.exists(knowledge_base_path):
            self.stderr.write(
                self.style.ERROR(f'Knowledge base directory does not exist: {knowledge_base_path}')
            )
            return
        
        # Initialize embeddings
        embeddings = self._initialize_embeddings()
        if not embeddings:
            self.stderr.write(
                self.style.ERROR('Failed to initialize embeddings. Check AWS credentials.')
            )
            return
        
        # ChromaDB persist directory
        persist_directory = os.path.join(settings.BASE_DIR, 'chroma_db')
        
        # Remove existing database if rebuild requested
        if options['rebuild']:
            import shutil
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory)
                self.stdout.write('Removed existing ChromaDB data')
        
        # Load documents
        self.stdout.write('Loading documents from knowledge base...')
        documents = self._load_documents(knowledge_base_path)
        
        if not documents:
            self.stderr.write(
                self.style.ERROR('No documents found in knowledge base')
            )
            return
        
        self.stdout.write(f'Found {len(documents)} documents')
        
        # Split documents
        self.stdout.write('Splitting documents into chunks...')
        splits = self._split_documents(documents)
        self.stdout.write(f'Created {len(splits)} document chunks')
        
        # Create or update ChromaDB vector store
        self.stdout.write('Creating ChromaDB vector store...')
        try:
            vector_store = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=persist_directory,
                collection_name="data_engineering_knowledge"
            )
            
            self.stdout.write(
                self.style.SUCCESS(
                    f'Successfully initialized ChromaDB knowledge base with {len(splits)} chunks'
                )
            )
            
            # Test the vector store
            self.stdout.write('Testing knowledge base search...')
            test_results = vector_store.similarity_search("data pipeline architecture", k=3)
            self.stdout.write(f'Test search returned {len(test_results)} results')
            
            if test_results:
                self.stdout.write('Sample result:')
                self.stdout.write(f'  Document: {test_results[0].metadata.get("source", "Unknown")}')
                self.stdout.write(f'  Content preview: {test_results[0].page_content[:200]}...')
            
        except Exception as e:
            self.stderr.write(
                self.style.ERROR(f'Failed to create ChromaDB vector store: {e}')
            )
            return
        
        self.stdout.write(
            self.style.SUCCESS('ChromaDB knowledge base initialization completed successfully!')
        )

    def _initialize_embeddings(self):
        """Initialize Hugging Face embeddings"""
        try:
            self.stdout.write('Initializing Hugging Face embeddings...')
            
            # Use a lightweight, efficient embedding model
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            
            # Initialize Hugging Face embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': True}
            )
            
            self.stdout.write(f'Successfully initialized embeddings with model: {model_name}')
            return embeddings
            
        except Exception as e:
            self.stderr.write(f'Error initializing embeddings: {e}')
            return None

    def _load_documents(self, knowledge_base_path):
        """Load documents from knowledge base directory"""
        try:
            loader = DirectoryLoader(
                knowledge_base_path,
                glob="**/*.md",
                loader_cls=TextLoader,
                recursive=True
            )
            documents = loader.load()
            
            # Add some metadata processing
            for doc in documents:
                # Extract category from file path
                relative_path = os.path.relpath(doc.metadata['source'], knowledge_base_path)
                category = relative_path.split(os.sep)[0] if os.sep in relative_path else 'general'
                doc.metadata['category'] = category
                doc.metadata['file_name'] = os.path.basename(doc.metadata['source'])
            
            return documents
            
        except Exception as e:
            self.stderr.write(f'Error loading documents: {e}')
            return []

    def _split_documents(self, documents):
        """Split documents into smaller chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        splits = text_splitter.split_documents(documents)
        
        # Add chunk metadata
        for i, split in enumerate(splits):
            split.metadata['chunk_id'] = i
            split.metadata['chunk_size'] = len(split.page_content)
        
        return splits
