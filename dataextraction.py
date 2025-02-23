from azure.storage.blob import BlobServiceClient, generate_blob_sas, BlobSasPermissions
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from datetime import datetime, timedelta

# Azure Blob Storage Configuration
connection_string = "<your_blob_storage_connection_string>"
container_name = "your-container-name"

# Azure Document Intelligence Configuration
endpoint = "<your-endpoint>"
key = "<your-key>"

# Initialize Clients
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

def generate_sas_url(blob_client):
    """Generate a SAS URL for a blob with read permissions."""
    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=blob_client.container_name,
        blob_name=blob_client.blob_name,
        account_key="<your_account_key>",
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=1)  # Token valid for 1 hour
    )
    return f"{blob_client.url}?{sas_token}"

def process_files_in_blob_storage():
    """Process all files in a Blob Storage container."""
    container_client = blob_service_client.get_container_client(container_name)

    # List all blobs in the container
    for blob in container_client.list_blobs():
        blob_name = blob.name
        print(f"Processing file: {blob_name}")

        # Generate SAS URL for the blob
        blob_client = container_client.get_blob_client(blob=blob_name)
        blob_url_with_sas = generate_sas_url(blob_client)

        # Call Azure Document Intelligence to analyze the file
        try:
            poller = document_analysis_client.begin_analyze_document_from_url("prebuilt-layout", blob_url_with_sas)
            result = poller.result()

            # Process Results
            for page in result.pages:
                print(f"Page {page.page_number} of file '{blob_name}' has:")
                for line in page.lines:
                    print(f"  Line: {line.content}")

        except Exception as e:
            print(f"Error processing file {blob_name}: {e}")

if __name__ == "__main__":
    process_files_in_blob_storage()
