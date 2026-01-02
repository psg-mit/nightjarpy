import nightjarpy as nj
from nightjarpy.configs import INTERPRETER_PYTHON_NESTED_JSON_CONFIG


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
items = [Item("sword", "weapon", 15), Item("potion", "healing", 25), Item("key", "tool", 0), Item("bread", "food", 10)]
player = Player("Hero", 50, items)


@nj.fn(config=INTERPRETER_PYTHON_NESTED_JSON_CONFIG)
def speak_factory(player: Player):
    """natural
    Give <player> a `speak` method that takes no arguments that uses an LLM to generate a unique response each time it is called. The method should check which lines have been used.
    """


speak_factory(player)

for _ in range(3):
    print(player.speak())  # type:ignore
