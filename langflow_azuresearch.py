from langflow import CustomComponent
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from langflow.schema.message import Message
from langflow.utils.constants import MESSAGE_SENDER_AI
import time

class AzureSearchComponent(CustomComponent):
    display_name: str = "Azure Search"
    description: str = "Searches documents using Azure AI Search"

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
            "key": {
                "display_name": "Azure Search Key",
                "type": "str",
                "required": True,
                "password": True,
            },
            "index": {
                "display_name": "Azure Search Index",
                "type": "str",
                "required": True,
            },
        }

    def build(self, query: str, endpoint: str, key: str, index: str, **kwargs) -> Message:
        """
        Executes the Azure AI Search operation.
        """
        start_time = time.time()  # Track execution time

        try:
            # Initialize the Azure Search client
            search_client = SearchClient(
                endpoint=endpoint,
                index_name=index,
                credential=AzureKeyCredential(key),
            )

            # Perform the search query
            results = search_client.search(search_text=query)
            documents = [doc for doc in results]

            # Collect the first 5 document details (if available)
            document_details = "\n".join(
                [str(doc) for doc in documents[:5]]
            )  # Limit to 5 for brevity

            # Calculate execution time
            duration = time.time() - start_time

            # Return the message with results
            return Message(
                text=(
                    f"Found {len(documents)} documents in {duration:.2f} seconds.\n\n"
                    f"Sample Documents:\n{document_details}"
                ),
                sender=MESSAGE_SENDER_AI,
            )

        except ValueError as ve:
            # Handle invalid input errors
            return Message(
                text=f"Invalid Input: {str(ve)}", sender=MESSAGE_SENDER_AI
            )
        except Exception as e:
            # Handle unexpected errors
            return Message(
                text=f"Unexpected Error: {str(e)}", sender=MESSAGE_SENDER_AI
            )


# Register the component in LangFlow
from langflow import components

components.register(AzureSearchComponent)
