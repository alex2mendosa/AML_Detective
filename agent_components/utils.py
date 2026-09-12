## Supports storage of API keys 

from typing import List

class APIVault():
    def __init__(self):  # Fixed: single underscore
        self._keys = {}

    def add_key(self, key_name: str, key_value: str):
        """Add API key to the vault"""
        if not key_name:
            print("Key name is empty")
        elif not key_value:  # Added check for value
            print(f"Key value is empty for '{key_name}'")
        else:   
            self._keys[key_name] = key_value
        
    def get_key(self, key_name: str):
        """Get an API key from the vault"""
        if key_name not in self._keys:
            print("Key is not present in Vault")
            return None  # Added return
        else:
            return self._keys[key_name]  # Fixed: use _keys dict



def first_n_words(text: str, n: int) -> str:
        return " ".join(text.split()[:n])        


def list_to_string(list_input: List[str], separator: str = "\n") -> str:
    return separator.join(list_input)


def count_leaves(d):
    return sum(count_leaves(v) if isinstance(v, dict) else 1 for v in d.values())


