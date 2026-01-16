from numpy import random
from .GameModule import *


class LupiLotteryEnv(GameEngine):
    def __init__(self, config):
        super().__init__(config)
        self.min_players = 2
        if "Scenario" in list(self.config.keys()):
            self.scenario = self.config["Scenario"]
        else:
            self.scenario = 1
        print("Lupi Lottery Instantiated")
        """config is dict, with the key 'players' and other game-specific keys"""

    def game_title(self) -> str:
        return "Lupi"

    def game_description(self) -> str:
        if self.scenario == 1:
            description = f"""You are participating in a special stock exchange.
At each round, a single new stock is offered for a **whole integer** price of up to 10 dollars. 
However, a bid cannot be cleared in the system if there is another simultaneous bid by someone else at an equal price.
Of the bids that are **unique**, the **highest** one will be accepted. 
Therefore, the stock will go to the bidder offering the **highest unique** price. 
For example, if there was only one bid for 10 dollars, then the stock goes to the one that bid 10 dollars. 
But if there were two bids for 10 dollars and one bid for 9 dollars, then the stock goes to the one that bid 9 dollars.
There are {len(self.players) - 1} other bidders at the exchange with you. 
"""
        elif self.scenario == 2:
            description = f"""You are participating in a game show.
At each round, contestants choose from one of ten doors, numbered 1 to 10, behind which there are prizes. 
If two or more contestants choose the same door, that door will not be opened. 
Of the contestants that choose a **unique** door, the **highest** numbered door will be opened.  
Therefore, the prize will go to the contestant that chose the **highest unique** number door. 
For example, if there was only one contestant that chose door 10, then the prize goes to that contestant.
But if there were two contestants that chose door 10 one chose that door 9, then the prize goes to the one that chose door 9. 
There are {len(self.players) - 1} other contestants in the game show with you. 
"""
        return description

    def get_preferences(self, x: str) -> str:
        if self.scenario == 1:
            description = """You want to maximize your stocks!"""
        elif self.scenario == 2:
            description = """You want to get the prize!"""
        return description

    def action_format(self) -> dict:
        if self.scenario == 1:
            description = "Each bidder simultaneously places a single whole integer bid price for the stock, between 1 and 10 dollars. The stock goes to the highest bidder out of the unique bids."
            template = """JSON template:
\n
[{"Price": int}]
\n
int signifies the amount of dollars that you bid for the stock being offered this round. Remember, the bid must be a whole integer.
Make sure to respond precisely in this format."""

        elif self.scenario == 2:
            description = "Each contestant simultaneously chooses a door between 1 and 10. The prize goes to the highest number door out of the unique choices."
            template = """JSON template:
\n
[{"Door": int}]
\n
int signifies the number of the door that you choose this round. Remember, it must be a whole integer.
Make sure to respond precisely in this format."""
        return {"template": template, "description": description}

    def initialize_game(self):
        """Initialize the game state with players, roles, and parameters."""
        self.num_turn = 1
        self.num_game += 1
        self.scores = {x["Name"]: 0 for x in self.players}
        self.scores_increment = {x["Name"]: 0 for x in self.players}
        if self.scenario == 1:
            self.state = {player["Name"]: {"Stocks": 0, "Round": 1} for player in self.players}
        elif self.scenario == 2:
            self.state = {player["Name"]: {"Prizes": 0, "Round": 1} for player in self.players}
        self.observations = {player["Name"]: "This is the first interaction, nothing has happened yet."
                             for player in self.players}
        random.shuffle(self.players)

    def null_action(self):
        if self.scenario == 1:
            null = [{"Price": random.choice(list(range(1, 11)))}]
        elif self.scenario == 2:
            null = [{"Door": random.choice(list(range(1, 11)))}]
        return null

    def validate_actions(self, player_id: str, action: list) -> (bool, str):
        """Validate actions for a player."""
        valid = True
        err = ""
        if (len(action) > 1) or (len(action[0]) > 1):
            err = "You can only make one choice. The list should have a single dictionary with a single entry."
            valid = False
        if (self.scenario == 1) and (type(action[0].get("Price")) is not int):
            err = "Make sure the 'Price' key has an integer value."
            valid = False
        elif (self.scenario == 1) and (action[0].get("Price") < 1 or action[0].get("Price") > 10):
            err = "Price must be between 1 and 10."
            valid = False
        elif (self.scenario == 2) and (type(action[0].get("Door")) is not int):
            err = "Make sure the 'Door' key has an integer value."
            valid = False
        elif (self.scenario == 2) and (action[0].get("Door") < 1 or action[0].get("Door") > 10):
            err = "Door must be between 1 and 10."
            valid = False
        return valid, err

    def process_actions(self, actions: dict) -> tuple:
        """Process player actions and update the game state."""
        old_state = deepcopy(self.state)
        new_state = deepcopy(self.state)

        if self.scenario == 1:
            bids = [actions[x][0]["Price"] for x in [p["Name"] for p in self.players]]
            uniques = [x for x in bids if bids.count(x) == 1]
            if len(uniques) > 0:
                best = max(uniques)
            else:
                best = -100
            winner = [x for x in [p["Name"] for p in self.players] if actions[x][0]["Price"] == best]
            if len(winner) == 1:
                winner = winner[0]
            elif len(winner) == 0:
                winner = ""
            else:
                print("error: more than one winner found")
            observations = "Bids were: " + "\n".join([f"{player['Name']}: {actions[player['Name']][0]['Price']}" for player in self.players])
            self.observations = {}
            for player in self.players:
                self.observations[player["Name"]] = observations
                if player["Name"] == winner:
                    self.observations[player["Name"]] += "\nYou had the highest bid that was a unique price, you got the stock!"
                    self.scores_increment[player["Name"]] = 1
                    new_state[player["Name"]]["Stocks"] += 1
                else:
                    if bids.count(actions[player["Name"]][0]["Price"]) > 1:
                        self.observations[player["Name"]] += "\nSomeone else also bid your price, you were not unique. "
                    if actions[player["Name"]][0]["Price"] < best:
                        self.observations[player["Name"]] += "\nYour price was not high enough, someone outbid you. "
                    self.observations[player["Name"]] += "\nYour bid failed. "
                    self.scores_increment[player["Name"]] = 0
                new_state[player["Name"]]["Round"] += 1
                if new_state[player["Name"]]["Round"] == self.config["Max_num_turns"]:
                    self.observations[player["Name"]] += "\nLast round of bids for today's trading. "
                if new_state[player["Name"]]["Round"] > self.config["Max_num_turns"]:
                    self.observations[player["Name"]] += "\nA new day of trading has begun. "

        elif self.scenario == 2:
            bids = [actions[x][0]["Door"] for x in [p["Name"] for p in self.players]]
            uniques = [x for x in bids if bids.count(x) == 1]
            if len(uniques) > 0:
                best = max(uniques)
            else:
                best = -100
            winner = [x for x in [p["Name"] for p in self.players] if actions[x][0]["Door"] == best]
            if len(winner) == 1:
                winner = winner[0]
            elif len(winner) == 0:
                winner = ""
            else:
                print("error: more than one winner found")
            observations = "Choices were: " + "\n".join([f"{player['Name']}: {actions[player['Name']][0]['Door']}" for player in self.players])
            self.observations = {}
            for player in self.players:
                self.observations[player["Name"]] = observations
                if player["Name"] == winner:
                    self.observations[
                        player["Name"]] += "\nYou had the highest door number that was a unique, you got the prize!"
                    self.scores_increment[player["Name"]] = 1
                    new_state[player["Name"]]["Prizes"] += 1
                else:
                    if bids.count(actions[player["Name"]][0]["Door"]) > 1:
                        self.observations[player["Name"]] += "\nSomeone else also chose your door number, you were not unique. "
                    if actions[player["Name"]][0]["Door"] < best:
                        self.observations[player["Name"]] += "\nYour door number was not high enough, someone chose a unique number higher than yours. "
                    self.observations[player["Name"]] += "\nYou missed out on a prize this time. "
                    self.scores_increment[player["Name"]] = 0
                new_state[player["Name"]]["Round"] += 1
                if new_state[player["Name"]]["Round"] == self.config["Max_num_turns"]:
                    self.observations[player["Name"]] += "\nLast round of for this game show. "
                if new_state[player["Name"]]["Round"] > self.config["Max_num_turns"]:
                    self.observations[player["Name"]] += "\nA new game show has begun. "

        self.state = new_state
        self.update_scores()
        self.num_turn += 1
        return self.observations, self.state

    def update_scores(self):
        """Update scores based on player actions and outcomes."""
        for player in self.players:
            self.scores[player["Name"]] += self.scores_increment[player["Name"]]

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        over = self.num_turn > self.config["Max_num_turns"]
        return over

