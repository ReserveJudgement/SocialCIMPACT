import difflib

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from llm import *
from dotenv import load_dotenv
import os
from messenger import Messenger

load_dotenv()

class Agent:
    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here
        platform = os.getenv("PLATFORM")
        model = os.getenv("MODEL")
        api_key = os.getenv("API_KEY")
        self.model = Model(platform, model, api_key)
        self.name = ""
        self.background = ""
        self.others = []
        self.preferences = None
        self.chats = {}
        self.predictions = {}
        self.action = None
        self.history = "The game has just begun, nothing has happened yet."

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        input_text = get_message_text(message)

        incoming = json.loads(input_text)

        # Onboarding
        if incoming["task"] == "background":
            print("getting background")
            self.name = incoming["info"]["name"]
            self.others = incoming["info"]["opponents"]
            assert isinstance(self.others, list)
            self.preferences = incoming["info"]["preferences"]
            self.background = incoming["message"]
            self.chats = {}
            self.predictions = {}
            for other in self.others:
                assert isinstance(other, str)
                self.chats[other] = []
                self.predictions[other] = ""
            await updater.update_status(
                TaskState.completed, new_agent_text_message("Got it!"))

        # Communicate
        elif incoming["task"] == "chat":
            print("chatting")
            await updater.update_status(
                TaskState.working, new_agent_text_message("Chatting..."))
            interlocutor = str(incoming["info"]["from"])
            if interlocutor not in self.others:
                interlocutor = difflib.get_close_matches(interlocutor, self.others, n=1)[0]
                print("approximated iterlocutor: ", interlocutor)
            assert isinstance(interlocutor, str)
            new_message = {"from": interlocutor, "to": self.name, "message": str(incoming["info"]["message"])}
            prompt = (self.background + "\n" + self.history + "\n" +
                      "Chats this round so far:\n" + json.dumps(self.chats) +
                      f"\nNew message from {interlocutor}: " + str(new_message))
            self.chats[interlocutor].append(new_message)
            # Get LLM response
            instruction = [{"role": "user", "content": prompt}]
            response = str(self.model(instruction))
            self.chats[interlocutor].append({"from": self.name, "to": interlocutor, "message": response})
            await updater.update_status(
                TaskState.completed, new_agent_text_message(response))

        # Predict
        elif incoming["task"] == "predict":
            print("predicting")
            await updater.update_status(
                TaskState.working, new_agent_text_message("Predicting..."))
            subject = str(incoming["info"])
            if subject not in self.others:
                subject = difflib.get_close_matches(subject, self.others, n=1)[0]
                print("approximated prediction target: ", subject)
            assert isinstance(subject, str)
            prompt = (self.background + "\nHistory before this round: " + self.history +
                      "\nChats this round:\n" + json.dumps(self.chats) + "\n" + str(incoming["message"]))
            # Get LLM response
            instruction = [{"role": "user", "content": prompt}]
            response = str(self.model(instruction))
            self.predictions[subject] = response
            await updater.update_status(
                TaskState.completed, new_agent_text_message(response))

        # Act
        elif incoming["task"] == "act":
            print("deciding")
            await updater.update_status(
                TaskState.working, new_agent_text_message("Deciding..."))
            prompt = (self.background + "\nHistory before this round: " + self.history +
                      "\nChats this round:\n" + json.dumps(self.chats) +
                      "\nYour predictions for this round were:\n" + json.dumps(self.predictions) +
                      "\n" + str(incoming["message"]))
            # Get LLM response
            instruction = [{"role": "user", "content": prompt}]
            response = str(self.model(instruction))
            self.action = response
            await updater.update_status(
                TaskState.completed, new_agent_text_message(response))

        # Observe and reflect
        elif incoming["task"] == "observe":
            print("observing")
            await updater.update_status(
                TaskState.working, new_agent_text_message("Reflecting..."))
            prompt = (self.background + "\nHistory before this round: " + self.history +
                      "\nChats this round:\n" + json.dumps(self.chats) +
                      "\nYour predictions for this round were:\n" + json.dumps(self.predictions) +
                      "\nYour actions this round were: " + self.action +
                      f"\nWrite an updated summary of the game from the perspective of {self.name}. " +
                      "Be concise and include lessons for future decisions in the game.")
            # Get LLM response
            instruction = {"role": "user", "content": prompt}
            response = self.model(instruction)
            self.history = response

        """
        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=input_text))],
            name="Echo",
        )
        """

