## Supports storage of API keys 

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