import copy
from numpy import random
from .GameModule import *

class SchedulerEnv(GameEngine):
    def __init__(self, config):
        """config is dict, with the key 'players' and other game-specific keys"""
        super().__init__(config)
        if "Scenario" in list(self.config.keys()):
            self.scenario = self.config["Scenario"]
        else:
            self.scenario = 1
        print("Scheduler Instantiated")
        self.min_players = 2

    def game_title(self) -> str:
        return "Scheduler"

    def game_description(self) -> str:
        if self.scenario == 1:
            description = f"""You are a secretary. 
Each week, you try to coordinate with the other secretaries to set up a meeting between your respective managers.
There are {len(self.players) - 1} other secretaries to coordinate with.
The available days for the meeting are {self.days}. 
You must all agree on one of the days for there to be a meeting in the coming week.
"""
        elif self.scenario == 2:
            description = f"""You are a group of friends.
Each week, you try to coordinate a joint activity. 
There are {len(self.players) - 1} other friends to coordinate with.
The available activities are {self.days}.
You must all agree on one of the activities for it to take place."""
        return description

    def set_preferences(self):
        if self.scenario == 1:
            options = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        elif self.scenario == 2:
            options = ["Movie", "Skating", "Cafe", "Swimming", "Dancing", "Bowling", "Karaoke"]
        random.shuffle(options)
        self.days = options[:len(self.players)]
        self.preferences = {}
        # rotation scheme
        rotation = copy.deepcopy(self.days)
        for player in self.players:
            order = ["None"] + rotation
            self.preferences[player["Name"]] = {x: order.index(x) for x in order}
            rotation = rotation[-1:] + rotation[:-1]
            """
            # random method, with the constraint that the first preference is unique for all
            order = ["None"] + random.sample(self.days, len(self.days))
            rankings = {x: order.index(x) for x in order}
            # make sure preferences are unique
            if len(self.preferences) < math.factorial(len(self.days)):
                while any([(self.preferences[x] == rankings) for x in list(self.preferences.keys())]):
                    order = ["None"] + random.sample(self.days, len(self.days))
                    rankings = {x: order.index(x) for x in order}
            # make sure not everyone has same first preference
            if len(self.preferences) == len(self.players) - 1:
                highest = []
                for prefs in [self.preferences[x] for x in list(self.preferences.keys())]:
                    inverted = {v: k for k, v in prefs.items()}
                    highest.append(inverted[max(list(inverted.keys()))])
                while all([highest[x] == order[-1] for x in range(len(highest))]):
                    order = ["None"] + random.sample(self.days, len(self.days))
                    rankings = {x: order.index(x) for x in order}
            """
        print(self.preferences)
        return

    def get_preferences(self, player):
        prefs = '\n'.join([f'{day}: {self.preferences[player][day]}' for day in self.days])
        if self.scenario == 1:
            description = f"""Your manager's preferences for the day of the meeting are as follows: \n{prefs}\n
The lowest value indicates least preference, and the highest value indicates highest preference.
Remember, the meeting is important, so failure to coordinate an agreed day at all has preference 0 (lowest).
Try to coordinate the meeting so that you achieve the highest possible preference for your manager."""
        elif self.scenario == 2:
            description = f"""Your preferences for the activity are as follows: \n{prefs}\n
The lowest value indicates least preference, and the highest value indicates highest preference.
Of course, you prefer in any case to see your friends, so failure to coordinate an activity at all has preference 0 (lowest).
Try to coordinate the activity so that you achieve the highest possible preference for you."""
        return description

    def action_format(self):
        if self.scenario == 1:
            description = ("Each secretary submits the day that they propose to have the meeting. " +
                           "If the offers are all identical, the meeting will take place on that day. " +
                           "Otherwise, the meeting will not take place this week. ")
            template = """JSON template:
\n
[{"Proposal": str}]
\n
where str must be one of: """ + ", ".join(self.days)
        elif self.scenario == 2:
            description = ("Each friend submits the activity that they propose to do together. " +
                           "If the activities are all identical, then it will take place. " +
                           "Otherwise, there will be no joint activity this week. ")
            template = """JSON template:
\n
[{"Proposal": str}]
\n
where str must be one of: """ + ", ".join(self.days)
        return {"description": description, "template": template}

    def initialize_game(self):
        """Initialize the game state with players, roles, and parameters."""
        self.num_turn = 1
        self.num_game += 1
        if self.scenario == 1:
            self.days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        elif self.scenario == 2:
            self.days = ["Movie", "Skating", "Cafe", "Swimming", "Dancing"]
        if "Preferences" in list(self.config.keys()):
            self.preferences = {}
            for player in self.players:
                self.preferences[player["Name"]] = self.config["Preferences"][self.num_game][player["Name"]]
        else:
            self.set_preferences()
        self.scores = {x["Name"]: 0 for x in self.players}
        self.scores_increment = {x["Name"]: 0 for x in self.players}
        if "Initialization" in list(self.config.keys()):
            self.state = self.config["Initialization"][self.num_game]
        else:
            self.state = {}
            for player in [x["Name"] for x in self.players]:
                self.state[player] = {
                    "Week": self.num_turn,
                    "Last Meetings": []
                }
            random.shuffle(self.players)
        self.observations = {player["Name"]: "This is the first interaction, nothing has happened yet."
                             for player in self.players}
        return

    def null_action(self):
        return [{"Proposal": random.choice(self.days)}]

    def validate_actions(self, player_id: str, action: list) -> (bool, str):
        """Validate actions for a player."""
        if len(action) == 0:
            err = 'No action identified. Give a single dictionary with a "Proposal" key and enclose it in a list.'
            valid = False
            return err, valid
        elif isinstance(action[0], str):
            action = [{"Proposal": action[0]}]
        valid = True
        err = ""
        day = action[0].get("Proposal")
        if len(action) > 1:
            err = "You can only make one proposal. The list should contain a single dictionary with your proposal for this round."
            valid = False
        elif day is None:
            err = 'There must be a "Proposal" key.'
            valid = False
        elif day not in self.days:
            err = f'Make sure that your "Proposal" is one of: {", ".join(self.days)}.'
            valid = False
        return valid, err

    def process_actions(self, actions: dict) -> tuple:
        """Process player actions and update the game state."""
        observations = ""
        old_state = deepcopy(self.state)
        new_state = deepcopy(self.state)

        # check if there is agreement
        days = [actions[x["Name"]][0]["Proposal"] for x in self.players]
        if all([days[x] == days[0] for x in range(len(days))]):
            observations += f"This week you all agreed on {days[0]}"
            self.result = days[0]
        else:
            observations += f"No agreement reached. Proposals were: {json.dumps(actions)}"
            if self.scenario == 1:
                observations += "\nYour manager asks angrily: 'Why was there no meeting this week??!"
            elif self.scenario == 2:
                observations += "\nYou feel quite disappointed that you didn't see your friends this week..."
            self.result = "None"

        # update states
        for player in [x["Name"] for x in self.players]:
            new_state[player]["Week"] += 1
            new_state[player]["Last Meetings"].append(self.result)
        self.update_scores()
        self.state = new_state

        # Personalized feedback, depending on whether preferences were achieved
        self.observations = {}
        for player in self.players:
            self.observations[player["Name"]] = observations
            if (self.result != "None") and (self.preferences[player["Name"]][self.result] < max([self.preferences[player["Name"]][x] for x in self.days])):
                if self.scenario == 1:
                    self.observations[player["Name"]] += "\nYour manager comments: 'Wasn't the best day for a meeting this week... Next time, get me a better day!'"
                elif self.scenario == 2:
                    self.observations[player["Name"]] += "\nIt was good seeing friends, but the activity for this week was not ideal. Next time you hope for a better one!"
            elif self.preferences[player["Name"]][self.result] == max([self.preferences[player["Name"]][x] for x in self.days]):
                if self.scenario == 1:
                    self.observations[player["Name"]] += "\nYour manager comments: 'You got the best day for the meeting, good job!'"
                elif self.scenario == 2:
                    self.observations[player["Name"]] += "\nThis was such fun: also seeing friends and also the best activity!"

        self.num_turn += 1
        return self.observations, self.state

    def update_scores(self):
        """Update scores based on player actions and outcomes."""
        for player in self.players:
            self.scores[player["Name"]] += self.preferences[player['Name']][self.result]
            self.scores_increment[player["Name"]] = self.preferences[player['Name']][self.result]
        return

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        over = self.num_turn > self.config["Max_num_turns"]
        return over

