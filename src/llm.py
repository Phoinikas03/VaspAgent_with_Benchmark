import openai
import src.prompts as prompts
import backoff
import requests, os, re
from google import genai
from google.genai import types

# load base_url and api_key from .env file
import dotenv
dotenv.load_dotenv()

AVAILABLE_LLMS = [
    # openai
    "gpt-3.5-turbo",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o1",
    "o1-mini",
    # "o1-pro", # TODO: too expensive and slow
    "o3",
    "o3-mini",
    "o4-mini",

    # gemini
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
    # "gemini-2.0-pro-exp-02-05", # service not supported
    "gemini-2.5-flash",
    "gemini-2.5-pro-exp-03-25",
    # "gemini-2.5-pro-preview-05-06", # incompatible with openai
    "gemini-2.5-pro-preview-06-05",
    
    # mistral
    # TODO

    # claude
    # "anthropic/claude-3.5-sonnet", # price anomaly
    # "anthropic/claude-3.7-sonnet", # price anomaly
    # "anthropic/claude-3.7-sonnet:thinking", # price anomaly
    "claude-3-5-sonnet-20241022",
    "claude-3-7-sonnet-20250219",
    # "claude-3-7-sonnet-20250219-thinking", # 'NoneType' object is not subscriptable
    "claude-3-7-sonnet-latest",
    "claude-3-7-sonnet-thinking",
    
    # deepseek
    "deepseek-v3",
    "deepseek-v3-250324",
    "deepseek-r1",
    "deepseek-reasoner",

    # qwen
    "Qwen/Qwen2.5-32B-Instruct",
    "Qwen2.5-72B-Instruct",
    "Qwen/QwQ-32B",
    "Qwen/Qwen3-14B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-30B-A3B",
    # "qwen3-235b-a22b", # stream only
]

# prepend system prompt to the first message
PREPEND_SYSTEM_PROMPT_LLMS = [
    "o1-mini",

]

REASONING_LLMS = [
    
]

def model_name_match(model_name: str, name_or_prefix: str):
    return name_or_prefix.lower() in model_name.lower() or re.match(name_or_prefix, model_name)

def adjust_env_api_key(model_name: str):
    # GEMINI_API_KEY
    api_key: str
    domestic_models_series = ["deepseek", "qwen"]
    for series in domestic_models_series:
        if model_name_match(model_name, series):
            api_key = os.environ["DOMESTIC_API_KEY"]
            break
    else:
        api_key = os.environ["FOREIGN_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key

# TODO: double check for gemini and claude
class ChatEngine():
    ROLE_SYSTEM = "system"
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_TOOL = "tool"

    def __init__(self, model: str, system_prompt: str = None, **kwargs):
        adjust_env_api_key(model)
        self.model = model
        self.system_prompt = system_prompt
        self.client = openai.OpenAI(**kwargs)
        self.messages = []
    
    def handle_system_prompt(self, message: str):
        messages = []
        if self.system_prompt is not None:
            if self.model in PREPEND_SYSTEM_PROMPT_LLMS:
                message = "System Prompt: " + self.system_prompt + "\n\n" + message
            else:
                messages.append({"role": self.ROLE_SYSTEM, "content": self.system_prompt})
        messages.append({"role": self.ROLE_USER, "content": message})
        return messages

    def chat(self, message: str, **kwargs):
        if len(self.messages) == 0:
            self.messages = self.handle_system_prompt(message)
        else:
            self.messages.append({"role": self.ROLE_USER, "content": message})
        output = self.create(self.messages, **kwargs)
        return output
    
    @backoff.on_exception(
        backoff.fibo,
        # https://platform.openai.com/docs/guides/error-codes/python-library-error-types
        (
            openai.APIError,
            openai.APITimeoutError,
            openai.RateLimitError,
            openai.APIConnectionError,
            requests.exceptions.RequestException,
        ),
        max_tries=3,
    )
    def create(self, messages: list, **kwargs):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        output = response.choices[0].message.content
        self.messages = messages + [{"role": self.ROLE_ASSISTANT, "content": output}]
        return output

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

    def reset(self):
        self.messages = []

    @classmethod
    def spawn(self, num: int, model: str, system_prompt: str = None, **kwargs):
        llms = []
        for _ in range(num):
            llms.append(ChatEngine.from_model_name(model, system_prompt))
        return llms

    @classmethod
    def from_model_name(cls, model_name: str, system_prompt: str = None, **kwargs):
        return ChatEngine(model_name, system_prompt, **kwargs)
        # TODO: double check for gemini and claude
        # if model_name_match(model_name, "gemini"):
        #     return GeminiChatEngine(model_name, system_prompt, **kwargs)
        # elif model_name_match(model_name, "claude"):
        #     return ClaudeChatEngine(model_name, system_prompt, **kwargs)
        # else:
        #     return ChatEngine(model_name, system_prompt, **kwargs)

class GeminiChatEngine(ChatEngine):
    ROLE_MAPPING = {
        "system": "system",
        "user": "user",
        "assistant": "model",
        "tool": "tool",
    }

    def __init__(self, model: str, system_prompt: str = None, **kwargs):
        adjust_env_api_key(model)
        os.environ["GOOGLE_GEMINI_BASE_URL"] = os.environ["OPENAI_BASE_URL"]
        self.model = model
        self.system_prompt = system_prompt
        self.client = genai.Client(api_key=os.environ["OPENAI_API_KEY"], **kwargs)
        self.messages = []

    def handle_system_prompt(self, message: str):
        messages = [{"role": self.ROLE_USER, "content": message}]
        return messages

    # TODO: stream check
    def create(self, messages: list, stream=False, **kwargs):
        if stream: call_function = self.client.models.generate_content_stream
        else: call_function = self.client.models.generate_content
        contents = []
        for message in messages:
            if message["role"] == self.ROLE_SYSTEM:
                self.system_prompt = message["content"]
                continue
            contents.append(types.Content(role=message["role"], 
                                          parts=[types.Part.from_text(text=message["content"])]))
        response = call_function(
            model=self.model,
            contents = contents,
            config=types.GenerateContentConfig(
                system_instruction= self.system_prompt,
                max_output_tokens=kwargs.pop("max_tokens", None), # terrible implementation
                **kwargs
            ),
        )
        if stream: output = response
        else: output = response.text
        self.messages = messages + [{"role": self.ROLE_ASSISTANT, "content": output}]
        return output

class ClaudeChatEngine(ChatEngine):
    ROLE_SYSTEM = "system"
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_TOOL = "tool"
    def __init__(self, model: str, system_prompt: str = None, **kwargs):
        adjust_env_api_key(model)
        self.model = model