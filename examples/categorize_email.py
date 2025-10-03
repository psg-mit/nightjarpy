import nightjarpy as nj


class Email:
    def __init__(self, subject: str, body: str, sender: str):
        self.subject = subject
        self.body = body
        self.sender = sender
        self.category = None
        self.priority = None

    def __str__(self):
        return f"Email: {self.subject} (Category: {self.category}, Priority: {self.priority})"


email = Email(
    subject="URGENT: Server down in production",
    body="The main database server has crashed and we're losing customers. Need immediate attention!",
    sender="ops@company.com",
)


@nj.fn
def categorize_email(email: Email):
    """natural
    Analyze the <email> content and automatically categorize it as one of: 'urgent', 'bug_report', 'feature_request', 'spam', or 'general'.
    Also determine priority level: 'high', 'medium', or 'low' based on urgency indicators.
    Update the email's category and priority attributes.
    """


categorize_email(email)
print(email)  # Email: URGENT: Server down in production (Category: urgent, Priority: high)
