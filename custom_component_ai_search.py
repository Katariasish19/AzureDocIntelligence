from langflow import CustomComponent
from langchain.vectorstores import AzureSearch
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langflow.schema.message import Message
from langflow.utils.constants import MESSAGE_SENDER_AI
import os
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO)

class AzureSearchLangChainComponent(CustomComponent):
    display_name: str = "Azure Search (LangChain)"
    description: str = "Searches documents using Azure Cognitive Search with LangChain integration."

    def build_config(self):
        """
        Defines the configuration for the Azure Search component.
        """
        return {
            "query": {
                "display_name": "Search Query",
                "type": "str",
                "required": True,
            },
            "endpoint": {
                "display_name": "Azure Search Endpoint",
                "type": "str",
                "required": True,
            },
            "index": {
                "display_name": "Azure Search Index",
                "type": "str",
                "required": True,
            },
            "max_results": {
                "display_name": "Max Results",
                "type": "int",
                "default": 5,
            },
        }

    def build(self, query: str, endpoint: str, index: str, max_results: int = 5, **kwargs) -> Message:
        """
        Executes the Azure AI Search operation using LangChain.
        """
        start_time = time.time()

        # Retrieve the Azure Search Key and OpenAI API Key from environment variables
        azure_search_key = os.getenv("AZURE_SEARCH_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not azure_search_key:
            return Message(
                text="Azure Search Key not found. Please set the 'AZURE_SEARCH_KEY' environment variable.",
                sender=MESSAGE_SENDER_AI,
            )

        if not openai_api_key:
            return Message(
                text="OpenAI API Key not found. Please set the 'OPENAI_API_KEY' environment variable.",
                sender=MESSAGE_SENDER_AI,
            )

        try:
            # Initialize the OpenAI embeddings
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

            # Initialize the Azure Cognitive Search vector store
            vector_store = AzureSearch(
                azure_search_endpoint=endpoint,
                azure_search_key=azure_search_key,
                index_name=index,
                embedding_function=embeddings.embed_query,
            )

            # Perform the search query
            results = vector_store.similarity_search(query=query, k=max_results)

            # Collect document details
            document_details = "\n".join(
                [f"Document {i + 1}: {result.page_content}" for i, result in enumerate(results)]
            )

            # Calculate execution time
            duration = time.time() - start_time

            # Log query execution
            logging.info(f"Query executed in {duration:.2f} seconds.")

            # Return the message with results
            return Message(
                text=(f"Found {len(results)} documents in {duration:.2f} seconds.\n\n"
                      f"Sample Documents:\n{document_details}"),
                sender=MESSAGE_SENDER_AI,
            )

        except ValueError as ve:
            # Handle invalid input errors
            logging.error(f"Invalid Input Error: {str(ve)}")
            return Message(
                text=f"Invalid Input: {str(ve)}", sender=MESSAGE_SENDER_AI
            )
        except Exception as e:
            # Handle unexpected errors
            logging.error(f"Unexpected Error: {str(e)}")
            return Message(
                text=f"Unexpected Error: {str(e)}", sender=MESSAGE_SENDER_AI
            )


# Register the component in LangFlow
from langflow import components

components.register(AzureSearchLangChainComponent)
