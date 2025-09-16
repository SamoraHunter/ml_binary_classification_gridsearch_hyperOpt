from typing import List


class ListHolder:
    """A class to hold a list of manually removed features."""

    def __init__(self) -> None:
        """Initializes the ListHolder with an empty feature list."""
        self.feature_list: List[str] = []
