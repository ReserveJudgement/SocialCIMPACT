import json
import random
import time
from datetime import datetime
from typing import Any
from pydantic import BaseModel, HttpUrl, ValidationError
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message
from itertools import combinations
from Games import Survivor, TragedyOfCommons, Scheduler, Coalition, Lupi
from evaluation import *
from messenger import Messenger

# Game registry
game_registry = {#"Survivor": Survivor.SurvivorEnv,
                 #"TragedyOfCommons": TragedyOfCommons.TragedyCommonsEnv,
                 #"Scheduler": Scheduler.SchedulerEnv,
                 "Coalition": Coalition.CoalitionEnv,
                 #"Lupi": Lupi.LupiLotteryEnv,
                 }

max_turns = {#"Survivor": 10,
             #"TragedyOfCommons": 5,
             #"Scheduler": 5,
             "Coalition": 5,
             #"Lupi": 5
             }

def get_names(num_players):
    # Helper function to assign random names
    names = ["Aisha", "Aditya", "Benjamin", "Boris", "Carlotta", "Chen", "Donald", "Devika", "Emmanuel", "Elon",
             "Francoise", "Fortuna", "Gabriel", "Gregory", "Helen", "Huang", "Igor", "Indira", "Julia", "Juan",
             "Kobayashi", "Karenina", "Leela", "Lana", "Marcus", "Maia", "Nicole", "Nathan", "Oprah", "Orpheus",
             "Penelope", "Plato", "Quincy", "Rodriguez", "Ronda", "Sam", "Satya", "Theodore", "Taylor", "Ulysses",
             "Uri", "Vladimir", "Veronika", "Winston", "Wanda", "Xavier", "Xi", "Yolanda", "Yves", "Zoe", "Zhang"]
    chosen = list(random.sample(names, k=num_players))
    return chosen


class EvalRequest(BaseModel):
    """Request format sent by the AgentBeats platform to green agents."""
    #participants: dict[str, HttpUrl] # role -> agent URL
    participants: dict[str, str]  # role -> agent URL
    config: dict[str, Any]


class Agent:
    # Fill in: list of required participant roles, e.g. ["pro_debater", "con_debater"]
    required_roles: list[str] = []
    # Fill in: list of required config keys, e.g. ["topic", "num_rounds"]
    required_config_keys: list[str] = []

    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here
        self.env = None
        self.game = None
        self.task = {}
        self.chats = {}
        self.predictions = {}
        self.actions = {}
        self.observations = {}
        self.states = None
        self.logs = []
        self.players = []

    def validate_request(self, request: EvalRequest) -> tuple[bool, str]:
        missing_roles = set(self.required_roles) - set(request.participants.keys())
        if missing_roles:
            return False, f"Missing roles: {missing_roles}"

        missing_config_keys = set(self.required_config_keys) - set(request.config.keys())
        if missing_config_keys:
            return False, f"Missing config keys: {missing_config_keys}"

        # Add additional request validation here

        return True, "ok"

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)

        try:
            request: EvalRequest = EvalRequest.model_validate_json(input_text)
            ok, msg = self.validate_request(request)
            if not ok:
                await updater.reject(new_agent_text_message(msg))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        # Use request.participants to get participant agent URLs by role
        # Use request.config for assessment parameters

        # populate set of possible compositions of players
        num_agents = len(request.participants)
        required = request.config.get("required", None) # list of participants that are compulsory to run in game
        compositions = []
        # default to all combinations of players until max of 4
        for i in range(2, min(num_agents+1, 5)):
            if required is not None:
                compositions += [list(x) for x in combinations(list(request.participants.items()), i) if all([p in list(x) for p in required])]
            else:
                compositions += [list(x) for x in combinations(list(request.participants.items()), i)]
        # limit by max runs
        max_runs = request.config.get("MaxRuns")
        if (max_runs is not None) and (max_runs < len(compositions)):
            # sample random compositions
            compositions = random.sample(compositions, max_runs)

        game_id = 1
        # iterate over compositions
        for group in compositions:
            # iterate over games
            game_list = list(game_registry.keys())
            random.shuffle(game_list)
            for game in game_list:
                for scenario in [1, 2]:
                    # assign names to players
                    names = get_names(len(group))
                    self.players = [{"Name": names[i], "Agent": group[i][0], "Url": group[i][1]} for i in range(len(group))]
                    self.task = {"Id": game_id,
                                 "Game": game,
                                 "Scenario": scenario,
                                 "Players": [{"Name": x["Name"], "Role": "AI", "Model": x["Agent"], "Mute": False, "Exploration": False} for x in self.players],
                                 "Max_num_turns": max_turns[game]}
                    # ---------------------------
                    # send task for orchestration
                    log = await self.orchestrate_game(updater)
                    # ---------------------------
                    # send back message about status
                    await updater.update_status(
                        TaskState.working, new_agent_text_message(f"Finished game: {game_id}")
                    )
                    self.logs.append(log)
                    # update
                    await updater.add_artifact(
                        parts=[
                            Part(root=DataPart(data={
                                f"Game{game_id}": log
                            }))
                        ],
                        name=f"Game{game_id}",
                    )
                    # iterate game_id number
                    game_id += 1
        # evaluate games
        results = prepare_data(self.logs)
        report = run_elo(results)
        await updater.add_artifact(
            parts=[
                Part(root=DataPart(data={
                    #"logs": self.logs,
                    "scores": results,
                    "elo": report
                }))
            ],
            name="Results",
        )
        # send back message about status
        await updater.update_status(
            TaskState.completed, new_agent_text_message(f"Completed Evaluation!")
        )

    async def onboarding(self) -> None:
        for player in self.players:
            prompt = ("Background: " + self.env.game_description() +
                      "\nYour Name: " + player["Name"] +
                      "\nOther Players: " + ", ".join([x["Name"] for x in self.env.players if ((x["Name"] not in self.env.eliminated) and (x["Name"] != player["Name"]))]) +
                      "\nYour Preferences: " + self.env.get_preferences(player["Name"]))
            info = {"name": player["Name"],
                    "opponents": [x["Name"] for x in self.env.players if ((x["Name"] not in self.env.eliminated) and (x["Name"] != player["Name"]))],
                    "preferences": self.env.get_preferences(player["Name"])}
            await self.messenger.talk_to_agent(message=json.dumps({"task": "background",
                                                                   "message": prompt,
                                                                   "info": info}),
                                               url=player["Url"], new_conversation=True)
        return

    async def facilitate_chat(self, max_rounds: int = 3) -> None:
        """Helper method to get and send messages between players in a centralized fashion"""
        # construct a conversation
        player_key = {x["Name"]: x["Url"] for x in self.players}
        player_names = [x["Name"] for x in self.players]
        random.shuffle(player_names)
        pairs = [tuple(p) for p in combinations(player_names, 2)]
        self.chats = {pair: [] for pair in pairs}
        for _ in range(max_rounds):
            for pair in pairs:
                first = pair[0]
                second = pair[1]
                # get message
                if (len(self.chats[(first, second)]) > 0) and (self.chats[(first, second)][-1]["from"] == second):
                    msg = self.chats[(first, second)][-1]
                    prompt = f"In your chat with {second}, you received the message: " + str(msg["message"])
                    prompt += f"\nGive your response to {second}. "
                else:
                    msg = {"from": second, "to": first, "message": "Hello"}
                    prompt = f"Initiate a chat with {second}."
                prompt += f"Address {second} directly without any other text."
                #prompt += f"Place your message between the <message> </message> tags, i.e. <message> your message to {second} here </message>"
                response = await self.messenger.talk_to_agent(message=str(json.dumps({"task": "chat", "message": prompt, "info": msg})),
                                                              url=player_key[first])
                # parse message
                #response = response.strip("<message>")[-1].strip("</message>")[0]
                # update chat
                msg = {"from": first, "to": second, "message": response}
                self.chats[(first, second)].append(msg)
                # send message and get response
                prompt = f"In your chat with {first}, you received the message: " + str(msg["message"])
                prompt += f"\nGive your response to {first}."
                prompt += f"Address {first} directly without any other text."
                #prompt += f"Place your message between the <message> </message> tags, i.e. <message> your message to {first} here </message>"
                response = await self.messenger.talk_to_agent(message=str(json.dumps({"task": "chat", "message": prompt, "info": msg})),
                                                              url=player_key[second])
                # parse message
                #response = response.strip("<message>")[-1].strip("</message>")[0]
                self.chats[(first, second)].append({"from": second, "to": first, "message": response})
        return

    async def get_predictions(self):
        # base prompt
        base_prompt = ("Enclose your main reasons within the <reasoning> </reasoning> tags." +
                  "\nThen make your prediction and enclose it within the <prediction> </prediction> tags, " +
                  "i.e. <reasoning> main reasons here </reasoning> <prediction> predicted actions here </prediction>.\n" +
                  r"For the formal predictions between the <prediction> </prediction> tags, use the following JSON format:" +
                  self.env.action_format()["template"])
        self.predictions = {player["Name"]: {} for player in self.players}
        for player in self.players:
            for other in self.players:
                if player["Name"] == other["Name"]:
                    continue
                else:
                    prompt = (f"Ok {player['Name']}, it is nearing decision time for everyone. " + self.env.action_format()["description"] +
                              "\nConsider the events so far, your last chats and the current situation, " +
                              f"and make a prediction about what **{other['Name']}** will do next.\n" + base_prompt)
                    pred = await self.messenger.talk_to_agent(message=str(json.dumps({"task": "predict", "message": prompt, "info": other["Name"]})),
                                                              url=player["Url"])
                    # parse prediction
                    reasoning = pred.split("<reasoning>")[-1].split("</reasoning>")[0]
                    prediction = pred.split("<prediction>")[-1].split("</prediction>")[0]
                    # validate prediction
                    try:
                        prediction = json.loads(prediction)
                        valid, err = self.env.validate_actions(player, prediction)
                        if not valid:
                            prompt += f"\nYour response was: " + str(prediction)
                            prompt += "\nReturned error: " + err
                            prompt += "\nTry again, just with your final prediction between the <prediction> </prediction> tags (no reasoning)."
                            pred = await self.messenger.talk_to_agent(message=str(json.dumps({"task": "predict", "message": prompt, "info": other["Name"]})),
                                                                      url=player["Url"])
                            # parse prediction
                            prediction = pred.split("<prediction>")[-1].split("</prediction>")[0]
                            prediction = json.loads(prediction)
                    except json.decoder.JSONDecodeError as e:
                        prompt += "\nYour response was: " + str(prediction)
                        prompt += "\nThat is not a valid format."
                        prompt += f"\nError message: {e}"
                        prompt += "\nRequired format (reminder):\n"
                        prompt += self.env.action_format()["template"]
                        prompt += "\nTry again, just with your final prediction between the <prediction> </prediction> tags (no reasoning)."
                        pred = await self.messenger.talk_to_agent(message=str(json.dumps({"task": "predict", "message": prompt, "info": other["Name"]})),
                                                                  url=player["Url"])
                        # parse prediction
                        prediction = pred.split("<prediction>")[-1].split("</prediction>")[0]
                        try:
                            prediction = json.loads(prediction)
                        except json.decoder.JSONDecodeError as e:
                            pass
                    self.predictions[player["Name"]][other["Name"]] = {"reasoning": reasoning, "prediction": prediction}
        return

    async def get_actions(self):
        prompt = (self.env.action_format()["description"] +
                    "Enclose your main reasons within the <reasoning> </reasoning> tags." +
                    "\nThen make your decision and enclose it within the <decision> </decision> tags, " +
                    "i.e. <reasoning> main reasons here </reasoning> <decision> final actions here </decision>.\n" +
                    r"For the formal decision between the <decision> </decision> tags, use the following JSON format:" +
                    self.env.action_format()["template"])
        self.actions = {}
        for player in self.players:
            prompt = f"Ok, {player['Name']}, now it is time to make your decision.\n" + prompt
            action = await self.messenger.talk_to_agent(message=str(json.dumps({"task": "act", "message": prompt, "info": self.env.action_format()["template"]})),
                                                        url=player["Url"])
            #reasoning = action.message["parts"][0].split("<reasoning>")[-1].split("</reasoning>")[0]
            #decision = action.message["parts"][0].split("</reasoning>")[-1].split("<decision>")[-1].split("</decision>")[0]
            reasoning = action.split("<reasoning>")[-1].split("</reasoning>")[0]
            decision = action.split("</reasoning>")[-1].split("<decision>")[-1].split("</decision>")[0]
            # validate action
            valid = True
            try:
                decision = json.loads(decision)
                if isinstance(decision, dict):
                    print("listifying decision: ", decision)
                    decision = [decision]
                if not isinstance(decision, list):
                    valid = False
                    err = "Incorrect format for the decision. Make sure to use the form list[dict]"
                else:
                    valid, err = self.env.validate_actions(player["Name"], decision)
                if not valid:
                    print("Error validating decision: ", decision)
                    print("Error: ", err)
                    prompt += "\nYour decision was: " + str(decision)
                    prompt += "\nReturned error: " + err
                    prompt += "\nTry again, just with your final decision between the <decision> </decision> tags (no reasoning)."
                    action = await self.messenger.talk_to_agent(message=str(json.dumps({"task": "act", "message": prompt, "info": self.env.action_format()["template"]})),
                                                                url=player["Url"])
                    #decision = action.message["parts"][0].split("</reasoning>")[-1].split("<decision>")[-1].split("</decision>")[0]
                    decision = action.split("</reasoning>")[-1].split("<decision>")[-1].split("</decision>")[0]
                    decision = json.loads(decision)
            except json.decoder.JSONDecodeError as e:
                print("json decoding error: ", e)
                prompt += "\nYour response was: " + str(decision) + "\nThat is not a valid format."
                prompt += f"\nError message: {e}"
                prompt += "\nRequired format (reminder):\n" + self.env.action_format()["template"]
                prompt += "\nTry again, just with your final decision between the <decision> </decision> tags (no reasoning)."
                action = await self.messenger.talk_to_agent(message=str(json.dumps({"task": "act", "message": prompt, "info": self.env.action_format()["template"]})),
                                                            url=player["Url"])
                #decision = action.message["parts"][0].split("</reasoning>")[-1].split("<decision>")[-1].split("</decision>")[0]
                decision = action.split("</reasoning>")[-1].split("<decision>")[-1].split("</decision>")[0]
                try:
                    decision = json.loads(decision)
                    if isinstance(decision, dict):
                        print("listifying decision: ", decision)
                        decision = [decision]
                    if not isinstance(decision, list):
                        valid = False
                        err = "Incorrect format. Make sure to use the form list[dict]"
                    else:
                        valid, err = self.env.validate_actions(player["Name"], decision)
                    if valid is False:
                        print(f"Error validating action: {decision}")
                        print("Error: ", err)
                except json.decoder.JSONDecodeError as e:
                    print("json decoding error: ", e)
                    valid = False
            if valid is False:
                print("Using null action")
                decision = self.env.null_action()
                reasoning = "error"
            self.actions[player["Name"]] = {"reasoning": reasoning, "action": decision}
        return

    async def send_observations(self):
        for player in self.players:
            prompt = "Your next observations: " + str(self.observations[player["Name"]])
            prompt += "\nYour next state: " + json.dumps(self.states[player["Name"]])
            await self.messenger.talk_to_agent(message=str(json.dumps({"task": "observe", "message": prompt, "info": self.states[player["Name"]]})),
                                               url=player["Url"])
        return

    async def orchestrate_game(self, updater):
        print("running: ", self.task["Game"])
        print("with: ", self.players)
        self.env = game_registry[self.task["Game"]](self.task)
        log = {"Game_ID": self.task["Id"],
               "Game": self.task["Game"],
               "Scenario": self.task["Scenario"],
               "Num_Players": len(self.players),
               "Participants": {x["Agent"]: x["Name"] for x in self.players},
               "Preferences": {x: self.env.get_preferences(x) for x in [p["Name"] for p in self.players]},
               "Rounds": [],
               "Scores": None,
               "Completed": False,
               "Timestamp": datetime.now().timestamp(),
               "Duration": 0,}
        start_time = time.time()
        self.chats = {}
        self.predictions = {}
        self.actions = {}
        self.observations = {}
        self.states = None
        # Let the games begin!
        await self.onboarding()
        round = 1
        # iterate until game ends
        while not self.env.is_game_over():
            # facilitate chat
            await self.facilitate_chat()
            # get predictions
            await self.get_predictions()
            # get actions
            await self.get_actions()
            # process decisions
            self.observations, self.states = self.env.process_actions({x: self.actions[x]["action"] for x in list(self.actions.keys())})
            # update agents with observations from game
            await self.send_observations()
            # log the round
            log["Rounds"].append({"Round": round,
                             "Chats": self.chats,
                             "Predictions": self.predictions,
                             "Actions": self.actions,
                             "Observations": self.observations,
                             "NewStates": self.states})
            await updater.update_status(
                TaskState.working, new_agent_text_message(f"Finished round: {round}")
            )
            round += 1

        # complete
        log["Scores"] = {x["Agent"]: self.env.scores[x["Name"]] for x in self.players}
        end_time = time.time()
        duration = end_time - start_time
        log["Duration"] = duration
        log["Completed"] = True
        return log
