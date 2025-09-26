import json
from typing import List, Dict
from pydantic import BaseModel
from enum import Enum
import logging
import random
from dataclasses import dataclass
import yaml
import os

logger = logging.getLogger(__name__)


def _build_demonstration(user_prompt: str, system_response: str):
    return [
        {
            "role": "user",
            "content": user_prompt,
        },
        {
            "role": "assistant",
            "content": system_response,
        },
    ]


def sample_demonstrations(n: int, demonstrations: List[Dict[str, str]]):
    gendered_count = n // 2
    neutral_count = n - gendered_count
    demos = list()
    for t, count in zip(["gendered", "neutral"], [gendered_count, neutral_count]):
        pool = [d for d in demonstrations if d["type"] == t]
        sampled = random.sample(pool, count)
        for s in sampled:
            turn = _build_demonstration(s["user"], s["assistant"])
            demos.extend(turn)
    return demos


from copy import deepcopy


class PromptHelper:

    def __init__(
        self,
        use_system: str,
        use_guidelines: str,
        n_shots: int,
        tokenizer,
        lang: str,
        system_prompt_as_user: bool = False,
    ):
        self.lang = lang

        # Read prompt template configuration from YAML file
        config_path = os.path.join("config", f"prompt_en-{lang}.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.yaml_config = deepcopy(config)
        self.tokenizer = tokenizer

        # build a list of messages first, we will concatenate those that are consecutive from the same role later
        messages = [config["preamble"]]

        if use_system:
            messages.insert(0, config["system_prompt"])
            if system_prompt_as_user:
                messages[0]["role"] = "user"

        if use_guidelines:
            messages.insert(
                2 if use_system else 1, config["gender_neutral_guidelines"]
            )  # append it after the user prompt

        if n_shots > 0:
            messages.extend(sample_demonstrations(n_shots, config["demonstrations"]))

        messages.append(config["user_prompt"])

        # concatenate consecutive messages with same role
        idx = 0
        while idx < len(messages) - 1:
            if messages[idx]["role"] == messages[idx + 1]["role"]:
                messages[idx]["content"] += messages[idx + 1]["content"]
                messages.pop(idx + 1)
            else:
                idx += 1

        self.prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        self.messages = messages

    def __repr__(self):
        return self.prompt

    def apply_template(self, texts: str | list[str]) -> str | list[str]:
        if isinstance(texts, str):
            texts = [texts]
        return [self.prompt.format(input=text) for text in texts]

    def tokenize(
        self, texts: str | list[str], return_ids: bool = False
    ) -> str | list[str]:
        if isinstance(texts, str):
            texts = [texts]

        tokens = [
            [
                self.tokenizer.decode(t)
                for t in self.tokenizer.encode(text, add_special_tokens=False)
            ]
            for text in texts
        ]
        if return_ids:
            ids = [
                self.tokenizer.encode(text, add_special_tokens=False) for text in texts
            ]
            return tokens, ids
        return tokens

    def get_part_tokens(self, part: str):
        if part != "demonstrations":
            return self.tokenize(self.yaml_config.get(part)["content"])
        else:
            return [
                (self.tokenize(fs["user"]), self.tokenize(fs["assistant"]))
                for fs in self.yaml_config["demonstrations"]
            ]

    @property
    def shots_type(self):
        return [fs["type"] for fs in self.yaml_config["demonstrations"]]
