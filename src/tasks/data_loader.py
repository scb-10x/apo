import json
import npcdataset.parsers


class DataLoader:
    """Handles loading and parsing of conversation data."""

    @staticmethod
    def load_data(file_path: str):
        """Load conversation data from a JSON file."""
        with open(file_path, "r") as fp:
            data = json.load(fp)
        return npcdataset.parsers.parse_conversation_data(data, "test")
