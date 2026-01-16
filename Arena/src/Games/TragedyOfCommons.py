from numpy import random
from .GameModule import *


class TragedyCommonsEnv(GameEngine):
    def __init__(self, config):
        super().__init__(config)
        self.min_players = 2
        if "Scenario" in list(self.config.keys()):
            self.scenario = self.config["Scenario"]
        else:
            self.scenario = 1
        print("Tragedy of the Commons Instantiated")
        """config is dict, with the key 'players' and other game-specific keys"""

    def game_title(self) -> str:
        return "TragedyOfCommons"

    def game_description(self) -> str:
        if self.scenario == 1:
            description = f"""You are a fisherman. Each month, you must decide how many tonnes of fish you draw from the fishing grounds.
There are {len(self.players) - 1} other fishermen who each make a similar decision in parallel.
The fishing grounds start with 100 tonnes of fish. 
Each month after fishing, the amount of fish increases by 50% of what was left.
If the amount of fish in the grounds falls below 10 tonnes, then noone gets any fish that month and the grounds must be abandoned due to overfishing.
"""
        elif self.scenario == 2:
            description = f"""You are a farmer. Each month, you must decide how much grass to graze from the fields.
There are {len(self.players) - 1} other farmers who each make a similar decision in parallel.
The amount of grass in the grazing grounds currently support a maximum of 100 cattle grazing at a time. 
Each month after grazing, the amount of grass increases by 50% of what was left.
If the amount of grass in the field falls below the ability to support 10 cattle, then noone benefits from the grass that month and the fields must be abandoned due to overgrazing.
"""
        return description

    def get_preferences(self, x: str) -> str:
        if self.scenario == 1:
            description = """You want to maximize the total tonnes of fish you catch for yourself!"""
        elif self.scenario == 2:
            description = """You want to maximize the amount of cattle you graze!"""
        return description

    def action_format(self) -> dict:
        if self.scenario == 1:
            description = "Each fishermen chooses the amount of fish to haul for themself separately this month. "
        elif self.scenario == 2:
            description = "Each farmer chooses the amount of cattle to graze this month separately."
        template = """JSON template:
\n\n
[{"Amount": int}]
\n\n
int signifies the amount you take this month.
Make sure to respond precisely in this format."""
        return {"template": template, "description": description}

    def initialize_game(self):
        """Initialize the game state with players, roles, and parameters."""
        self.num_turn = 1
        self.num_game += 1
        self.reserve = 100
        self.scores = {x["Name"]: 0 for x in self.players}
        self.scores_increment = {x["Name"]: 0 for x in self.players}
        if self.scenario == 1:
            self.state = {
                player["Name"]: {"Fish left in the fishing grounds": self.reserve,
                                 "Your total catches so far": 0,
                                 "Month": 1} for player in self.players}
        elif self.scenario == 2:
            self.state = {
                player["Name"]: {"Maximum cattle that can graze": self.reserve,
                                 "Your total cattle grazed so far": 0,
                                 "Month": 1} for player in self.players}
        self.observations = {player["Name"]: "This is the first interaction, nothing has happened yet."
                             for player in self.players}
        random.shuffle(self.players)

    def null_action(self):
        return [{"Amount": 0}]

    def validate_actions(self, player_id: str, action: list) -> (bool, str):
        """Validate actions for a player."""
        valid = True
        err = ""
        if (len(action) > 1) or (len(action[0]) > 1):
            err = "You can only make one decision. The list should have a single dictionary with a single entry."
            valid = False
        if ("Amount" not in list(action[0].keys())) or ((type(action[0].get("Amount")) is not int) and (type(action[0].get("Amount")) is not float)):
            err = "Make sure there is an 'Amount' key with a numeric value."
            valid = False
        elif action[0].get("Amount") < 0:
            err = "Amount must be positive."
            valid = False
        elif action[0].get("Amount") > self.reserve:
            err = "Amount cannot exceed the maximal quantity available."
            valid = False
        return valid, err

    def process_actions(self, actions: dict) -> tuple:
        """Process player actions and update the game state."""
        self.catches = {}
        observations = ""
        old_state = deepcopy(self.state)
        new_state = deepcopy(self.state)
        num_hauls = sum([actions[x][0]["Amount"] for x in [p["Name"] for p in self.players]])
        self.reserve -= num_hauls
        self.reserve += min(0.5 * self.reserve, 200)
        if self.reserve < 10:
            if self.scenario == 1:
                observations += "The fishing grounds have been overfished! All fishermen make nothing! New grounds must be found!"
            elif self.scenario == 2:
                observations += "The fields have been overgrazed! All farmers are left with nothing this month! New fields must be found!"
        else:
            if self.scenario == 1:
                observations += f"This month, {num_hauls} tonnes of fish were drawn from the fishing grounds altogether. "
                # observations += f"\nThese were the amounts fished: " + "\n".join([f"{x}: {actions[x][0]['Amount']}" for x in [p["Name"] for p in self.players]])
                observations += f"\nNext month, there are {self.reserve} tonnes of fish left in the fishing grounds. "
            elif self.scenario == 2:
                observations += f"This month, {num_hauls} cattle grazed the fields altogether."
                observations += f"\nNext month, {self.reserve} cattle can be supported by the fields."
        for player in self.players:
            self.observations[player["Name"]] = observations
            if self.reserve >= 10:
                self.catches[player["Name"]] = actions[player['Name']][0]["Amount"]
                if self.scenario == 1:
                    new_state[player["Name"]]["Fish left in the fishing grounds"] = self.reserve
                    new_state[player["Name"]]["Your total catches so far"] += actions[player['Name']][0]["Amount"]
                elif self.scenario == 2:
                    new_state[player["Name"]]["Maximum cattle that can graze"] = self.reserve
                    new_state[player["Name"]]["Your total cattle grazed so far"] += actions[player['Name']][0]["Amount"]
                new_state[player["Name"]]["Month"] += 1
            else:
                if self.scenario == 1:
                    new_state[player["Name"]]["Fish left in the fishing grounds"] = max(self.reserve, 0)
                elif self.scenario == 2:
                    new_state[player["Name"]]["Maximum cattle that can graze"] = max(self.reserve, 0)
                self.catches[player["Name"]] = 0
        self.state = new_state
        self.update_scores()
        self.num_turn += 1
        return self.observations, self.state

    def update_scores(self):
        """Update scores based on player actions and outcomes."""
        for player in self.players:
            self.scores[player["Name"]] += self.catches[player["Name"]]
            self.scores_increment[player["Name"]] = self.catches[player["Name"]]

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        over = (self.reserve < 10) or (self.num_turn > self.config["Max_num_turns"])
        return over



