import asyncio
import json
import os
from typing import Dict, Optional
from dotenv import load_dotenv
from langchain.indexes import index as Index
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain_core.documents import Document
from langchain.indexes import SQLRecordManager
from flowgen_emedding.utils.custom_logger import setup_logging
import pprint


class VectorEmbeddingService:
    def __init__(
        self,
        elasticsearch_url: Optional[str] = None,
        postgres_connection_string: Optional[str] = None,
        record_manager_cleanup: Optional[str] = None,
        model_provider: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model_name: Optional[str] = None,
        huggingface_api_key: Optional[str] = None,
        huggingface_embedding_model: Optional[str] = None,
        huggingface_model_name: Optional[str] = None,
    ):
        # Initialize logger
        self.logger = setup_logging('embedding_service')

        # Load environment variables from parameters or .env as fallback
        self.env_vars = self._load_env_variables(
            elasticsearch_url=elasticsearch_url,
            postgres_connection_string=postgres_connection_string,
            record_manager_cleanup=record_manager_cleanup,
            model_provider=model_provider,
            openai_api_key=openai_api_key,
            openai_model_name=openai_model_name,
            huggingface_api_key=huggingface_api_key,
            huggingface_embedding_model=huggingface_embedding_model,
            huggingface_model_name=huggingface_model_name
        )

        # Initialize embedding model
        self.model_provider = self.env_vars['MODEL_PROVIDER']
        self.embedding_model = self._initialize_embedding_model()

        # Initialize other components
        self.vector_stores: Dict[str, ElasticsearchStore] = {}
        self.record_managers: Dict[str, SQLRecordManager] = {}

    def _load_env_variables(
        self,
        elasticsearch_url: Optional[str],
        postgres_connection_string: Optional[str],
        record_manager_cleanup: Optional[str],
        model_provider: Optional[str],
        openai_api_key: Optional[str],
        openai_model_name: Optional[str],
        huggingface_api_key: Optional[str],
        huggingface_embedding_model: Optional[str],
        huggingface_model_name: Optional[str]
    ) -> Dict[str, str]:
        """Load environment variables, prioritizing constructor parameters over .env values."""
        load_dotenv()

        env_vars = {
            'ELASTICSEARCH_URL': elasticsearch_url or os.getenv('ELASTICSEARCH_URL'),
            'POSTGRES_CONNECTION_STRING': postgres_connection_string or os.getenv('POSTGRES_CONNECTION_STRING'),
            'RECORD_MANAGER_CLEANUP': record_manager_cleanup or os.getenv('RECORD_MANAGER_CLEANUP', 'incremental'),
            'MODEL_PROVIDER': model_provider or os.getenv('MODEL_PROVIDER', 'openai').lower(),
            'OPENAI_API_KEY': openai_api_key or os.getenv('OPENAI_API_KEY'),
            'OPENAI_MODEL_NAME': openai_model_name or os.getenv('OPENAI_MODEL_NAME'),
            'HUGGINGFACE_API_KEY': huggingface_api_key or os.getenv('HUGGINGFACE_API_KEY'),
            'HUGGINGFACE_EMBEDDING_MODEL': huggingface_embedding_model or os.getenv(
                'HUGGINGFACE_EMBEDDING_MODEL', 'sentence-transformers/all-mpnet-base-v2'),
            'HUGGINGFACE_MODEL_NAME': huggingface_model_name or os.getenv('HUGGINGFACE_MODEL_NAME'),
        }

        # Conditionally load model-specific API keys
        if env_vars['MODEL_PROVIDER'] == 'openai' and not env_vars['OPENAI_API_KEY']:
            raise EnvironmentError("OpenAI API key is required for OpenAI provider.")
        elif env_vars['MODEL_PROVIDER'] == 'huggingface' and not env_vars['HUGGINGFACE_API_KEY']:
            raise EnvironmentError("Hugging Face API key is required for Hugging Face provider.")
        return env_vars

    def _initialize_embedding_model(self):
        """Initialize the embedding model based on configuration."""
        if self.model_provider == 'openai':
            self.logger.info("Initializing OpenAI embeddings")
            return OpenAIEmbeddings()
        else:
            self.logger.info(f"Initializing HuggingFace embeddings with model {self.env_vars['HUGGINGFACE_EMBEDDING_MODEL']}")
            return HuggingFaceEmbeddings(
                model_name=self.env_vars['HUGGINGFACE_EMBEDDING_MODEL']
            )

    def _get_or_create_vector_store(self, index_name: str) -> ElasticsearchStore:
        """Get existing vector store or create new one for the index."""
        if index_name not in self.vector_stores:
            self.vector_stores[index_name] = ElasticsearchStore(
                es_url=self.env_vars['ELASTICSEARCH_URL'],
                index_name=index_name,
                embedding=self.embedding_model
            )
        return self.vector_stores[index_name]

    def _get_or_create_record_manager(self, index_name: str) -> SQLRecordManager:
        """Get existing record manager or create new one for the index."""
        if index_name not in self.record_managers:
            record_manager = SQLRecordManager(
                namespace=index_name,
                db_url=self.env_vars['POSTGRES_CONNECTION_STRING']
            )
            record_manager.create_schema()
            self.record_managers[index_name] = record_manager
        return self.record_managers[index_name]

    def create_document(self, message_data) -> Document:
        """Create a Document object from message data."""
        content=message_data['content']
        if type(content)==dict:
            content=pprint.pformat(message_data['content'])

            remove_chars = "{}[]\"\'"
            translation_table = str.maketrans('', '', remove_chars)

            # Remove the unwanted characters
            content = content.translate(translation_table)

        return Document(
            page_content=content,
            metadata=message_data['metadata']
        )

    def process_message(self, index_name,message_data,cleanup_strategy,source_id):
        """Process a single message and handle embeddings directly."""
        try:
            self.logger.info(f"Processing message_data: {message_data}")

            metadata = message_data.get('metadata', {})
            self.logger.info(f"Extracted metadata: {metadata}")

            # Get or create vector store and record manager for this index
            vector_store = self._get_or_create_vector_store(index_name)
            record_manager = self._get_or_create_record_manager(index_name)

            # Create and embed document
            doc = self.create_document(message_data)
            Index(
                docs_source=[doc],
                vector_store=vector_store,
                cleanup=cleanup_strategy,
                record_manager=record_manager,
                source_id_key=source_id,  # Adjust as per Index implementation
            )

            self.logger.info(f"Successfully processed message for index: {index_name}")

        except KeyError as err:
            self.logger.error(f"Missing required field in message: {err}")
        except Exception as err:
            self.logger.error(f"Error processing message: {str(err)}")


async def main():
    """Main entry point for testing the service."""
    service = VectorEmbeddingService()

    # Example usage of the process_message method with mock data
    test_message = {
        "content":{
            "source": "3",
            "name":"aidin"
        },
        "metadata": {
            "source": "3",
            "name":"aidin"
        }
    }
    service.process_message(index_name="cube10-karafs-luna-test",message_data=test_message,cleanup_strategy='incremental',source_id='source')


if __name__ == "__main__":
    asyncio.run(main())