import nightjarpy as nj


@nj.fn
def filter_and_process(items: list[str]):
    valid_emails = []
    for item in items:
        """natural
        Check if <item> is a valid email address.
        If it's not a valid email, continue to the next loop iteration.
        If it is valid, add it to <valid_emails> list.
        """
    return valid_emails


emails = ["user@example.com", "invalid-email", "admin@company.org", "not-an-email", "support@help.com"]
valid = filter_and_process(emails)
print(f"Found {len(valid)} valid emails: {valid}")
