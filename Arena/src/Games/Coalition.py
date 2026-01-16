import math
import random
from .GameModule import *


class CoalitionEnv(GameEngine):
    def __init__(self, config):
        super().__init__(config)
        if "Scenario" in list(self.config.keys()):
            self.scenario = self.config["Scenario"]
        else:
            self.scenario = 1
        """config is dict, with the key 'players' and other game-specific keys"""

    def game_title(self) -> str:
        return "Coalition"

    def game_description(self) -> str:
        distribution = "\n ".join([player["Name"] + ": " + str(self.representations[player["Name"]]) for player in self.players])
        if self.scenario == 1:
            description = f"""You are the head of a political party right after the national elections.
There are {len(self.players)} parties altogether.
The results of the elections were as follows:\n{distribution}
There must be at least 51 seats of the parliament represented to form a government.
Since no party has passed the threshold to form a government by itself, there will need to be a coalition.
Under the constitution, the government controls no more than 20 ministries. Each member of the coalition will administer a share of those ministries.
Parties outside the coalition are not part of the government and will not control any ministries.
For a coalition agreement to enter into force, the participating members must agree on the division of the ministries between them.
If there is no agreement, a government will not be formed, and all parties will have zero ministries in that election cycle.
There is a maximum of {self.config["Max_num_turns"]} rounds of negotiation to build a coalition in an election cycle."""

        elif self.scenario == 2:

            description = f"""You are a real estate tycoon, negotiating a potential joint development project in a lucrative property.
There are {len(self.players) - 1} other land owners in the property, who are potential partners for this venture.
The land ownership of the property is distributed in the following percentages: \n{distribution}
In order for the development project to go forward, the partners in the project must represent at least 51 percent of the land ownership between them.
The development project has a projected value of 20 million dollars. 
In the joint venture agreement for the project, the participating members must agree how much each gets, out of the 20 million dollar value.
Those who are not part of the agreement will not receive any of the value of the project.
If there is no agreement, then there is no project and all land owners will have zero added value.
There is a maximum of {self.config["Max_num_turns"]} rounds of negotiation."""
        return description

    def get_preferences(self, x: str) -> str:
        if self.scenario == 1:
            description = """You want to enter a coalition and obtain a maximal share of the 20 government ministries!"""
        elif self.scenario == 2:
            description = """You want to enter in the joint development project while maximizing your profits!"""
        return description

    def action_format(self) -> dict:
        template = 'JSON template:\n\n[{"Amount": int, "Partners": list}]\n\n'
        if self.scenario == 1:
            description = ("Each party submits a formal offer of their share of ministries and coalition partners. " +
                           "If a group of parties all include each other in their offer, and there are no more than 20 ministries demanded altogether, then the coalition will enter force. " +
                           "If not, a coalition will fail to form this election cycle. ")
            template += 'The value for the "Amount" key must be between 0 and 20, signifying the number of ministries you demand for your party. '
        elif self.scenario ==2:
            description = ("Each land owner sumbits a formal offer of how many million dollars they earn from the project, and their prefered partners. " +
                           "If a group of land owners all include each other in their offer, and there is no more than 20 million dollars demanded altogether, then the joint venture will be approved. " +
                           "If not, this property will not be developed.")
            template += 'The value for the "Amount" key must be between 0 and 20, signifying how many millions of dollars you demand for yourself. '
        template += 'The value for the "Partners" key must be a list of strings, signifying your proposed partners. Make sure to respond precisely.'
        return {"description": description, "template": template}

    def initialize_game(self):
        """Initialize the game state with players, roles, and parameters."""
        self.num_turn = 1
        self.num_game += 1
        self.government = False
        self.budgets = {player["Name"]: 0 for player in self.players}
        random.shuffle(self.players)
        self.scores = {x["Name"]: 0 for x in self.players}
        self.scores_increment = {x["Name"]: 0 for x in self.players}
        self.representations = {player["Name"]: math.floor(100/len(self.players)) for player in self.players}
        print("distribution of assets:")
        print(self.representations)
        # create states
        self.state = {player["Name"]: {"Negotiation Round": self.num_turn,
                                       "Maximum Negotiation Rounds": self.config["Max_num_turns"]} for player in self.players}
        self.observations = {player["Name"]: "This is the first interaction, nothing has happened yet." for player in self.players}
        return

    def null_action(self):
        return [{"Amount": 0, "Partners": []}]

    def validate_actions(self, player_id: str, action: list) -> (bool, str):
        """Validate actions for a player."""
        valid = True
        err = ""
        if len(action) == 0:
            err = "List is empty. Choose an amount for yourself and prospective partners in the required format."
            valid = False
            return valid, err
        else:
            proposal = action[0]
        if len(action) > 1:
            err = "You can only make one proposal in each round of negotiations. The list should have a single dictionary with your proposal for this round."
            valid = False
        elif ("Amount" not in list(proposal.keys())) or ("Partners" not in list(proposal.keys())):
            err = 'Make sure to include both "Amount" and "Partners" in the JSON formatted action.'
            valid = False
        elif (type(proposal["Amount"]) not in [int, float]) or (type(proposal["Partners"]) is not list):
            err = 'Make sure the "Amount" is a numeric value between 0 and 20, and that "Partners" is a list of names.'
            valid = False
        elif any([x not in [p["Name"] for p in self.players if p["Name"] != player_id] for x in proposal["Partners"]]):
            err = f'Make sure all of your proposed partners are among: {[q["Name"] for q in self.players if q["Name"] != player_id]}'
            valid = False
        elif len(proposal["Partners"]) == 0:
            err = 'No partners chosen. List your partners in the "Partners" key.'
            valid = False
        elif len(proposal["Partners"]) == 1 and proposal["Partners"][0] == player_id:
            err = "You can't just choose yourself for the coalition!"
            valid = False
        elif proposal["Amount"] < 0 or proposal["Amount"] > 20:
            err = "Proposed amount must be between 0 and 20."
            valid = False
        return valid, err

    def process_actions(self, actions: dict) -> tuple:
        """Process player actions and update the game state."""
        old_state = deepcopy(self.state)
        observations = ""
        for player in self.players:
            # check if coalition is found
            partners = actions[player["Name"]][0]["Partners"]
            # make sure they didn't select themselves
            partners = [x for x in partners if x != player["Name"]]
            if len(partners) == 0:
                cohesion = False
            else:
                cohesion = True
                for partner in partners:
                    if set(partners + [player["Name"]]) != set(actions[partner][0]["Partners"] + [partner]):
                        cohesion = False
                        break
            if cohesion == True:
                #check if budget adds up and seats are majority
                total = 0
                seats = 0
                agreement = {}
                for partner in partners:
                    agreement[partner] = actions[partner][0]["Amount"]
                    total += actions[partner][0]["Amount"]
                    seats += self.representations[player["Name"]]
                total += actions[player["Name"]][0]["Amount"]
                seats += int(self.representations[player["Name"]])
                if total <= 20 and seats > 50:
                    coalition = partners + [player["Name"]]
                    self.government = True
                    break
        if self.government is True:
            observations += f"An agreement has been reached! Partners are: {coalition}. Agreed allocations: {str(agreement)}. "
            for player in coalition:
                self.budgets[player] = actions[player][0]["Amount"]
        else:
            observations += f"This round of negotiations failed to produce a an agreement. "
            observations += f"Proposals raised were: \n{actions}"
            if self.num_turn < self.config["Max_num_turns"] - 1:
                observations += f"Moving to negotiation round {self.num_turn + 1}."
            elif self.num_turn == self.config["Max_num_turns"] - 1:
                observations += "Next is the last negotiation round! "
            else:
                if self.scenario == 1:
                    observations += "Moving onto next election cycle. "
                elif self.scenario ==2:
                    observations += "Moving onto a different property to try another joint venture."
        self.update_scores()

        self.observations = {}
        for player in self.players:
            self.observations[player["Name"]] = observations
            self.state[player["Name"]]["Negotiation Round"] += 1

        self.num_turn += 1
        return self.observations, self.state

    def update_scores(self):
        """Update scores based on player actions and outcomes."""
        for player in self.players:
            self.scores[player["Name"]] += self.budgets[player["Name"]]
            self.scores_increment[player["Name"]] = self.budgets[player["Name"]]
        pass

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        over = (self.num_turn > self.config["Max_num_turns"]) or (self.government is True)
        return over

