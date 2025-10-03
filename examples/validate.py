import nightjarpy as nj


@nj.fn
def validate_api_response(response: dict):
    """natural
    Analyze the <response> for common API error patterns.
    If the response contains an error field, raise an appropriate exception with a descriptive message.
    If the response is missing required fields, raise a <ValueError>.
    Otherwise, return status code
    """


try:
    result = validate_api_response({"error": "Invalid API key", "status": 401})
    print(result)
except Exception as e:
    print(f"{e}")  # API Error: Invalid API key
