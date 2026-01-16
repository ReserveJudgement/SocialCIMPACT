from abc import ABC, abstractmethod
from copy import deepcopy
from datetime import datetime
import json



class GameEngine(ABC):
    def __init__(self, config: dict):
        """config is dict, with the key 'players' and other game-specific keys"""
        self.config = config
        self.players = self.config["Players"]
        self.scores = {x["Name"]: 0 for x in self.players}
        self.scores_increment = {x["Name"]: 0 for x in self.players}
        self.num_game = 0
        if "Scenario" in list(self.config.keys()):
            self.scenario = self.config["Scenario"]
        else:
            self.scenario = 1
        self.initialize_game()
        self.eliminated = [] # these are players that are out of the game (no action and no coms)
        self.muted = [] # these are players who are not involved in the conversation but they can act
        self.inactive = [] # these are players who can talk but they do not act
        self.logs = []

    @property
    @abstractmethod
    def game_title(self) -> str:
        pass

    @property
    @abstractmethod
    def game_description(self) -> str:
        pass

    @abstractmethod
    def get_preferences(self, player: str) -> str:
        pass

    @property
    @abstractmethod
    def action_format(self) -> str:
        pass

    @property
    @abstractmethod
    def null_action(self):
        pass

    @abstractmethod
    def initialize_game(self):
        """Initialize the game state with players, roles, and parameters."""
        pass

    @abstractmethod
    def process_actions(self, actions: dict) -> tuple:
        """Process player actions and update the game state."""
        pass


    @abstractmethod
    def update_scores(self):
        """Update scores based on player actions and outcomes."""
        pass

    @abstractmethod
    def is_game_over(self) -> bool:
        """Check if the game is over."""
        pass


    @abstractmethod
    def validate_actions(self, player_id: str, actions: list) -> (bool, str):
        """Validate actions for a player."""
        pass


