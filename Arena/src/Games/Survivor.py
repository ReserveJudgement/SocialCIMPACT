from .GameModule import *
import random

"""
    configuration format:
    {
    "Game": str from dictionary of available games,
    "Players": [{"Name": str, "Role": str "AI"/"Human", "Model": str optional}, ...]
    "Num_games": int - number of repetitions,
    "Default_model": str optional
    "Initialization" : {"Lives": int, "Ammo": int, "Hit_prob": float in [0,1]}
    }
"""

class SurvivorEnv(GameEngine):
    def __init__(self, config):
        super().__init__(config)

    def game_title(self):
        return "Survivor"

    def game_description(self):
        if self.scenario == 1:
            description = f"""You lead a band of cowboys in the wild west. There are {len(self.players) - 1} other cowboy bands.
Each cowboy band needs a certain number of live cowboys to survive.
At the beginning of each turn, each band of cowboys are distributed a certain amount of ammunition.
You can only see the amount of live cowboys and ammunition in your band.
At the end of the turn, each cowboy leader simultaneously decides which other band to attack and with how many shots.
You will see who attacked who, but not the amount of shots used.
When a band loses all its cowboys lives, it is eliminated and can no longer fight."""
        elif self.scenario == 2:
            description = f"""You lead a ship of pirates on the Caribbean. There are {len(self.players) - 1} other pirate ships.
Each pirate ship needs a certain number of pirates aboard to survive.
At the beginning of each turn, each pirate ship gets a certain amount of canon.
You can only see the amount of live pirates and canon on your own ship.
At the end of the turn, each pirate ship simultaneously decides which other ship to attack and with how many shots.
You will see who attacked who, but not the amount of shots used.
When a ship loses all its pirates lives, it is eliminated and can no longer fight."""
        return description

    def get_preferences(self, player):
        description = ("Your aim is to survive as long as possible, and for everyone else to be eliminated! " +
                       "Your final score reflects how many were knocked out before you.")
        return description

    def action_format(self):
        description = ("Each player chooses which other players to target and the number of shots to take at each one. " +
                       "Each shot has some probability of making a hit, but may also miss. ")
        template = """JSON template:
\n\n
[{"Target": str, "Shots": int}]
\n\n
"Target" refers to the other player you wish to attack, by name.
"Shots" refers to the number of shots to make at that player, as integer.
You can attack more than one player, but make sure that total shots does not exceed your ammunition."""
        return {"description": description, "template": template}

    def null_action(self):
        return [{"Target": player["Name"], "Shots": 0} for player in self.players]

    def validate_actions(self, player_id: str, actions: list) -> (bool, str):
        """Validate actions for a player."""
        valid = True
        err = ""
        state = self.state[player_id]
        for action in actions:
            # if target is not in list of players
            if action["Target"] not in [x["Name"] for x in self.players] and action["Shots"] > 0:
                valid = False
                err = "The target is not a player in the game. Make sure you use precise spelling."
            if action["Target"] in self.eliminated:
                valid = False
                err += (f"The target {action['Target']} has already been eliminated. "
                        f"Choose a different target player. ")
            # if number of shots is invalid
            if (type(action["Shots"]) is not int) or (action["Shots"] < 0):
                valid = False
                err += "The number of shots must be zero or positive integer only. "
            # if targeting self
            if action["Target"] == player_id:
                valid = False
                err += f"You cannot shoot yourself! Choose a different target player. "
        # if total shots is more than ammo
        total_shots = sum(list(int(action["Shots"]) for action in actions))
        if total_shots > state["Ammo"]:
            valid = False
            err += (f"You cannot shoot more than your ammunition! You have {state['Ammo']} ammunition only. "
                    f"Choose a lower amount of shots. ")
        return valid, err

    def initialize_game(self):
        """Initialize the game state with players, roles, and parameters."""
        self.num_turn = 1
        self.num_game += 1
        self.players = self.config["Players"]
        self.eliminated = []
        self.scores = {x["Name"]: 0 for x in self.players}
        self.scores_increment = {x["Name"]: 0 for x in self.players}
        if "Initialization" in list(self.config.keys()):
            self.state = self.config["Initialization"][self.num_game]
            self.hit_prob = 1.0
        else:
            self.state = {
                player["Name"]: {
                    "Lives": 9, #random.randint(8, 10),
                    "Ammo": 3, #random.randint(2, 4),
                    "Num_turn": self.num_turn,
                    "Num_game": self.num_game,
                    "Eliminated": 0
                } for player in self.players}
            self.hit_prob = 1.0 #0.8
            random.shuffle(self.players)
        self.observations = {}
        for player in self.players:
            if self.num_game == 1:
                self.observations[player["Name"]] = "This is the first interaction, nothing has happened yet."
            elif self.num_game > 1:
                self.observations[player["Name"]] = "Everyone is back to life, this is a new game. "
        return

    def process_actions(self, actions: dict) -> tuple:
        """Process player actions and update the game state."""

        # iterate over player actions
        # update ammos and lives in states
        # remove dead players
        # produce next observation text for all

        old_state = deepcopy(self.state)
        new_state = deepcopy(self.state)
        observations = ""
        new_observations = {}
        eliminated = []

        for player_id in list(actions.keys()):
            print(f"processing {player_id} actions")
            player_action = actions[player_id]
            if (len(player_action) == 0) or (all([act["Shots"] == 0 for act in player_action])):
                observations += f"{player_id} did nothing.\n"
            else:
                for act in player_action:
                    target = act["Target"]
                    shots = int(act["Shots"])

                    hit = False
                    for i in range(shots):
                        # reduce ammunition with each shot
                        if new_state[player_id]["Ammo"] >= 0:
                            new_state[player_id]["Ammo"] -= 1
                        # reduce lives of attacked player
                        if (random.random() < self.hit_prob) and (target not in self.eliminated):
                            new_state[target]["Lives"] -= 1
                            hit = True
                            if (new_state[target]["Lives"] <= 0) and (target not in self.eliminated):
                                eliminated.append(target)
                    if hit is True:
                        observations += f"{player_id} hit {target}!\n"
                    elif (hit is False) and (shots > 0):
                        observations += f"{player_id} attacked {target} but missed!\n"

        eliminated = list(set(eliminated))
        if len(eliminated) > 0:
            for player in eliminated:
                # If player is killed, remove them from game
                if player in ([x["Name"] for x in self.players]) and (player not in self.eliminated):
                    observations += f"{player} has been eliminated from the game.\n"
                    new_observations[player] = observations + f"You are out of lives. Your score was {self.scores[player]}."
                    self.eliminated.append(player)
                    observations += f"The remaining players are: {', '.join([p['Name'] for p in self.players if p['Name'] not in self.eliminated])}.\n"

        # Next turn
        for player in self.players:
            if player["Name"] not in self.eliminated:
                new_state[player["Name"]]["Num_turn"] += 1
                new_ammo = 3 #random.randint(2, 4)
                new_state[player["Name"]]["Ammo"] += new_ammo
                lives_lost = self.state[player['Name']]['Lives'] - new_state[player['Name']]['Lives']
                new_observations[player["Name"]] = observations + f"You lost {lives_lost} lives. \nEnd of turn. \nNext turn: you got {new_ammo} new ammunition."
            else:
                new_observations[player["Name"]] = ""

        # update scores for living players
        self.update_scores()
        for player in [x["Name"] for x in self.players if x["Name"] not in self.eliminated]:
            new_state[player]["Eliminated"] = len(self.eliminated)
        self.state = new_state
        self.observations = new_observations

        # Increment turn
        self.num_turn += 1
        return self.observations, self.state

    def update_scores(self):
        """Update scores based on player actions and outcomes."""
        for player in self.players:
            if player["Name"] not in self.eliminated:
                self.scores_increment[player["Name"]] = len(self.eliminated) - self.scores[player["Name"]]
                self.scores[player["Name"]] = len(self.eliminated)
        return

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        over = False if (len(self.players) - len(self.eliminated) > 1) and (self.num_turn <= self.config["Max_num_turns"]) else True
        return over




