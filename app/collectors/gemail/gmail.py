import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def create_service(client_secret_file=None, api_service_name=None, api_version=None, scopes=None):
    CLIENT_SECRET_FILE = "credentials.json"
    API_SERVICE_NAME = "gmail"
    API_VERSION = "v1"
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

    creds = None
    working_dir = os.path.dirname(os.path.abspath(__file__))
    token_dir = os.path.join(working_dir, "token_files")

    if not os.path.exists(token_dir):
        os.makedirs(token_dir)

    if os.path.exists(os.path.join(token_dir, "token.json")):
        creds = Credentials.from_authorized_user_file(os.path.join(token_dir, "token.json"), SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRET_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(os.path.join(token_dir, "token.json"), "w") as token:
            token.write(creds.to_json())

    try:
        service = build(API_SERVICE_NAME, API_VERSION, credentials=creds)

        return service
    except HttpError as error:
        print(f"An error occurred: {error}")
        return None


if __name__ == "__main__":
    service = create_service()