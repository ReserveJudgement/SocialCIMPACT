import gc
import json
import time
import openai
from openai import OpenAI
import requests
from google import genai
from google.genai import types
import os
from ollama import Client
import lmstudio as lms

class Model:
    def __init__(self, providor="LMStudio", model=None, api_key=None, rpm=None, tpm=None, max_tokens=1000):
        self.providor = providor
        self.model = model
        self.rpm = rpm
        self.tpm = tpm
        self.max_tokens = max_tokens
        self.num_requests = 0
        self.tokens_used = 0

        if providor == "Google":
            if api_key is None:
                print("Error: Google model needs an API key")
                exit()

            self.llm = genai.Client(api_key=api_key)

        elif providor == "OpenAI":
            self.llm = OpenAI()

        elif providor == "Ollama":
            self.llm = Client(
                host='http://localhost:11434',
                headers={'x-some-header': 'some-value'})

        elif providor == "LMStudio":
            self.llm = lms.llm(self.model)

        elif providor == "OpenRouter":
            self.llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    def __call__(self, prompt):
        if self.providor == "Google":
            messages = json.dumps(prompt)
            for i in range(3):
                if self.rpm is not None:
                    time.sleep((60 / self.rpm) + 0.2)
                self.num_requests += 1
                now = time.time()
                expected_token_use = self.model.models.count_tokens(model=self.model, contents=messages).total_tokens
                self.tokens_used += expected_token_use
                if (self.rpm is not None) and (self.num_requests >= self.rpm) and (now - self.last_time <= 60):
                    time.sleep(60 - (now - self.last_time))
                    self.num_requests = 0
                # get response
                try:
                    response = self.llm.models.generate_content(model=self.model,
                                                                  contents=messages,
                                                                  config=types.GenerateContentConfig(safety_settings=self.safety_config))
                    # check token usage also after generation
                    self.tokens_used += response.usage_metadata.total_token_count
                    if (self.tpm is not None) and (self.tokens_used >= self.tpm) and (now - self.last_time <= 60):
                        time.sleep(60 - (now - self.last_time))
                        self.tokens_used = 0
                    self.last_time = time.time()
                    break
                except Exception as e:
                    print(f"Error: ", e)
                    if i == 2:
                        return f"Error: {e}"
                    else:
                        time.sleep(5)
            return response.text

        elif self.providor == "OpenAI":
            for _ in range(3):
                try:
                    response = self.llm.responses.create(
                        model=self.model,
                        input=prompt,
                        #reasoning={"effort": "medium"},
                        #text={"verbosity": "medium"}
                        )
                    response = response.output_text
                    break
                except openai.RateLimitError as e:
                    response = f"Error: {e}"
                    print(response)
                    time.sleep(60)
                except Exception as e:
                    response = f"Error: {e}"
                    print(response)
                    time.sleep(5)
            return response

        elif self.providor == "Ollama":
            text = ""
            for i in range(3):
                try:
                    response = self.llm.chat(model=self.model_name, messages=prompt) # for thinking models think=False...
                    text = response.message.content
                    break
                except Exception as e:
                    print(f"Error: {self.model}: {e}")
                    if i < 2:
                        time.sleep(2)
                    else:
                        text = f"Error: {e}"
            return text

        elif self.providor == "LMStudio":
            return str(self.llm.respond(str(json.dumps(prompt))))

        elif self.providor == "OpenRouter":
            response = None
            for i in range(5):
                self.num_requests += 1
                if self.rpm is not None:
                    time.sleep((60 / self.rpm) + 0.2)
                self.num_requests += 1
                now = time.time()
                if (self.rpm is not None) and (self.num_requests >= self.rpm) and (now - self.last_time <= 60):
                    time.sleep(60 - (now - self.last_time))
                    self.num_requests = 0
                    self.last_time = time.time()
                # get response
                try:
                    response = self.llm.chat.completions.create(model=self.model, messages=prompt)
                except Exception as e:
                    print("Exception while getting request")
                    print("Response: ", response)
                    print("Error: ", str(e))
                    if i < 4:
                        time.sleep(5)
                    elif i == 4:
                        return "Error: " + str(e)

                if isinstance(response, dict) and ("error" in response.keys()):
                    err_msg = f"Error: {str(response['error'])}"
                    print("Error in response")
                    print(str(response))
                    print(err_msg)
                    if i < 4:
                        time.sleep(5)
                    elif i == 4:
                        return err_msg
                else:
                    return response.choices[0].message.content












