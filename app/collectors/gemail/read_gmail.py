from app.collectors.gemail.gmail_utils import init_gmail_service, get_email_messages, get_email_message_details, \
    download_attachments_parent

service = init_gmail_service()

messages = get_email_messages(service, max_results=2)

for msg in messages:
    details = get_email_message_details(service, msg['id'])
    if details:
        print(f"Subject: {details['subject']}")
        print(f"Sender: {details['sender']}")
        print(f"Recipients: {details['recipients']}")
        print(f"Date: {details['date']}")
        print(f"Body: {details['body'][:100]}...")
        print(f"Has Attachments: {details['has_attachments']}")
        print(f"Starred: {details['star']}")
        print(f"Labels: {details['label']}")
        print("-" * 40)
        if details['has_attachments']:
            download_attachments_parent(service, 'me', msg['id'], 'attachments')
            print("Attachments found and downloaded.")



