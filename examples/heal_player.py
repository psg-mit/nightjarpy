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
items = [Item("sword", "weapon", 15), Item("potion", "healing", 25), Item("key", "tool", 0), Item("bread", "food", 10)]
player = Player("Hero", 50, items)


@nj.fn
def use_heal_item(player: Player):
    for item in player.inventory:
        """natural
        Check if <item> can be used to heal the player.
        If this item can heal, break out of the loop.
        """
    player.health += item.strength
    player.inventory.remove(item)
    print(f"Used {item.name}! Health: {player.health}")


use_heal_item(player)
