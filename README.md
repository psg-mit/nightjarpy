<div align="center">
<img src="./assets/nightjar2.png" alt="logo" width="400px" margin="10px"></img>

[![PyPI](https://img.shields.io/pypi/v/nightjarpy)](https://pypi.org/project/nightjarpy)
![PyPI - Downloads](https://img.shields.io/pypi/dm/nightjarpy)
[![license](https://img.shields.io/github/license/psg-mit/nightjarpy.svg)](https://github.com/psg-mit/nightjarpy/tree/main/LICENSE)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/psg-mit/nightjarpy)](https://github.com/psg-mit/nightjarpy/issues)

</div>


| :exclamation:  Warning: This library performs arbitrary code execution, which is very dangerous. Use at your own risk.    |
|----------------------------------------------|

Nightjar enables embedding <i>natural code</i>---code written in natural language---in Python programs with <i>shared program state</i>.
This means the natural code can read and write Python variables, read and write to Python objects, and implement control flow in your program.

## Installation

```bash
pip install nightjarpy
```

### Installing Research Dependencies

```bash
pip install nightjarpy[research]
```

### Installing Dev Dependencies

```bash
pip install nightjarpy[dev]
```

### Docker Container

You can also run Nightjar in a Docker container for consistent environments:

```bash
# Build the Docker image
docker build -t nightjarpy .

# Run the container
docker run -it nightjarpy
```

## LLM API
Nightjar currently supports OpenAI and Anthropic models as the backend LLM. Set your API keys in a `.env` file.
```
OPENAI_API_KEY=<your_api_key>
ANTHROPIC_API_KEY=<your_api_key>
```

## Quickstart

### Basic Usage

Nightjar allows you to write natural language code that integrates seamlessly with Python:

```python
import nightjarpy as nj


@nj.fn
def calculate_average(numbers):
    """natural
    Consider the values of <numbers> and compute the semantic average as <:result>
    """
    return result

result = calculate_average([1, "2", "three", "cuatro", "五"])
print(result)  # 3.0

```

### Object Manipulation

Natural code in Nightjar can work with Python objects and classes:

```python
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
    sender="ops@company.com"
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
```

### Control Flow

Natural language code supports Python control structures including breaking loops, continuing loops, and raising errors:

#### Breaking Loops

```python
import nightjarpy as nj

class Item:
    def __init__(self, name: str, item_type: str, strength: int = 0):
        self.name = name
        self.item_type = item_type
        self.strength = strength

class Player:
    def __init__(self, name: str, health: int, inventory: list[Item]):
        self.name = name
        self.health = health
        self.inventory = inventory

# Create items and player using Python
items = [
    Item("sword", "weapon", 15),
    Item("potion", "healing", 25),
    Item("key", "tool", 0),
    Item("bread", "food", 10)
]
player = Player("Hero", 50, items)

@nj.fn
def use_heal_item(player: Player):
    for item in player.inventory:
        """natural
        Check if <item> can be used to heal the player.
        If this item can heal, break out of the loop.
        """
    player.health += healing_item.strength
    player.inventory.remove(healing_item)
    print(f"Used {healing_item.name}! Health: {player.health}")

use_heal_item(player)
```

#### Continuing Loops

```python
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
```

#### Raising Errors

```python
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
```

### Configuration

You can configure the LLM backend and other settings:

```python
import nightjarpy as nj

# Use a different LLM model and temperature
config = nj.DEFAULT_CONFIG
config.llm = nj.LLMConfig(model="openai/gpt-5.1", temperature=0)

@nj.fn(config=config)
def complex_calculation(data):
    """natural
    Find the outliers in <data> and save as as list in <:outliers>
    """
    return outliers
```

## Syntax

Nightjar uses a simple syntax for embedding natural language in Python code:

### Function Decorators

Use the `@nj.fn` decorator to create functions with natural language implementations:

```python
import nightjarpy as nj

@nj.fn
def function_name(parameters):
    """natural
    Your natural language description here.
    Use <variable_name> to reference variables.
    Use <:result_variable> to assign to new variables.
    """
    return result_variable
```

### Variable References

- `<variable_name>` - Reference existing Python variables
- `<:new_variable>` - Create new variables