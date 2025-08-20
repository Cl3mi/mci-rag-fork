import os
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

from app.collectors.gemail.gmail import create_service


def init_gmail_service():
    """
    Initializes the Gmail service using the credentials stored in token.json.

    Returns:
        service: An authorized Gmail API service instance.
    """
    return create_service()


def create_email_message(sender, to, subject, message_text, attachments=None):
    """
    Creates an email message with optional attachments.

    Args:
        sender (str): The email address of the sender.
        to (str): The email address of the recipient.
        subject (str): The subject of the email.
        message_text (str): The body text of the email.
        attachments (list, optional): List of file paths to attach to the email.

    Returns:
        str: The raw email message encoded in base64url format.
    """
    message = MIMEMultipart()
    message["to"] = to
    message["from"] = sender
    message["subject"] = subject

    # Add the body text
    message.attach(MIMEText(message_text, "plain"))

    # Add attachments if any
    if attachments:
        for file_path in attachments:
            part = MIMEBase("application", "octet-stream")
            with open(file_path, "rb") as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename={os.path.basename(file_path)}")
            message.attach(part)

    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
    return raw_message


def send_email(service, sender, to, subject, message_text, attachments=None):
    """
    Sends an email using the Gmail API.

    Args:
        service: The authorized Gmail API service instance.
        sender (str): The email address of the sender.
        to (str): The email address of the recipient.
        subject (str): The subject of the email.
        message_text (str): The body text of the email.
        attachments (list, optional): List of file paths to attach to the email.

    Returns:
        dict: The response from the Gmail API after sending the email.
    """
    raw_message = create_email_message(sender, to, subject, message_text, attachments)
    message = {"raw": raw_message}

    try:
        response = service.users().messages().send(userId="me", body=message).execute()
        print(f"Message Id: {response['id']}")
        return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def _extract_body(payload):
    body = '<Text body not available>'
    if 'parts' in payload:
        for part in payload['parts']:
            if part['mimeType'] == 'multipart/alternative':
                for subpart in part['parts']:
                    if subpart['mimeType'] == 'text/plain' and 'data' in subpart['body']:
                        body = base64.urlsafe_b64decode(subpart['body']['data']).decode('utf-8')
                        break
            elif part['mimeType'] == 'text/plain' and 'data' in part['body']:
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                break
    elif 'body' in payload and 'data' in payload['body']:
        body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
    return body


def get_email_messages(service, user_id='me', label_ids=None, folder_name='INBOX', max_results=5):
    messages = []
    next_page_token = None

    # Resolve label ID from folder name (e.g., 'INBOX', 'SENT', etc.)
    if folder_name:
        label_results = service.users().labels().list(userId=user_id).execute()
        labels = label_results.get('labels', [])
        folder_label_id = next((label['id'] for label in labels if label['name'].lower() == folder_name.lower()), None)

        if folder_label_id:
            if label_ids:
                label_ids.append(folder_label_id)
            else:
                label_ids = [folder_label_id]
        else:
            raise ValueError(f"Folder '{folder_name}' not found.")

    while True:
        result = service.users().messages().list(
            userId=user_id,
            labelIds=label_ids,
            maxResults=min(500, max_results - len(messages)) if max_results else 500,
            pageToken=next_page_token
        ).execute()

        messages.extend(result.get('messages', []))
        next_page_token = result.get('nextPageToken')

        if not next_page_token or (max_results and len(messages) >= max_results):
            break

    return messages[:max_results] if max_results else messages


def get_email_message_details(service, msg_id):
    message = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    payload = message['payload']
    headers = payload.get('headers', [])

    subject = next((header['value'] for header in headers if header['name'].lower() == 'subject'), None)
    if not subject:
        subject = message.get('subject', 'No subject')

    sender = next((header['value'] for header in headers if header['name'] == 'From'), 'No sender')
    recipients = next((header['value'] for header in headers if header['name'] == 'To'), 'No recipients')

    snippet = message.get('snippet', 'No snippet')

    has_attachments = any(
        part.get('filename') for part in payload.get('parts', []) if part.get('filename')
    )

    date = next((header['value'] for header in headers if header['name'] == 'Date'), 'No date')
    star = message.get('labelIds', []).count('STARRED') > 0
    label = ', '.join(message.get('labelIds', []))

    body = _extract_body(payload)

    return {
        'subject': subject,
        'sender': sender,
        'recipients': recipients,
        'body': body,
        'snippet': snippet,
        'has_attachments': has_attachments,
        'date': date,
        'star': star,
        'label': label
    }


def download_attachments_parent(service, user_id, msg_id, target_dir):
    message = service.users().messages().get(userId=user_id, id=msg_id).execute()
    for part in message['payload'].get('parts', []):
        if part.get('filename'):
            att_id = part['body'].get('attachmentId')
            if att_id:
                att = service.users().messages().attachments().get(userId=user_id, messageId=msg_id,
                                                                   id=att_id).execute()
                data = att['data']
                file_data = base64.urlsafe_b64decode(data.encode('UTF-8'))
                file_path = os.path.join(target_dir, part['filename'])
                print('Saving attachment to:', file_path)
                with open(file_path, 'wb') as f:
                    f.write(file_data)


def download_attachments_all(service, user_id, msg_id, target_dir):
    thread = service.users().threads().get(userId=user_id, id=msg_id).execute()
    for message in thread.get('messages', []):
        for part in message['payload'].get('parts', []):
            if part.get('filename'):
                att_id = part['body'].get('attachmentId')
                if att_id:
                    att = service.users().messages().attachments().get(userId=user_id, messageId=message['id'],
                                                                       id=att_id).execute()
                    data = att['data']
                    file_data = base64.urlsafe_b64decode(data.encode('UTF-8'))
                    file_path = os.path.join(target_dir, part['filename'])
                    print('Saving attachment to:', file_path)
                    with open(file_path, 'wb') as f:
                        f.write(file_data)
