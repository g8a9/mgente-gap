import csv
import outlines
import argparse
import logging
import json
import os
import pandas as pd
from pandas import Series
from transformers import AutoTokenizer
from typing import Dict, List, Optional, Union, Iterable
from outlines.samplers import GreedySampler

# from vllm import LLM, SamplingParams
# from vllm.sampling_params import GuidedDecodingParams
import torch
from itertools import product
from tqdm import tqdm

# from pydantic import BaseModel
# from enum import Enum


class Prompt:
    """
    A class for few-shot prompts.
    :param user_template: template for `user` role messages
    :param assistant_template: template for *assistant* role messages
    :param system_message: (optional) *system* role prompt (**not all models support this**)
    :param shots_data: data to be used to replace placeholders in user and assistant role messages,
    in `dict` format with placeholders as keys and the corresponding content as values.
    """

    def __init__(
        self,
        user_template: str = None,
        assistant_template: str = None,
        shots_data: Optional[Union[pd.DataFrame, List[Dict[str, str]]]] = None,
        system_message: Optional[str] = None,
        message_list: List[Dict[str, str]] = None,
    ):
        self.system_message = system_message
        self.message_list = message_list
        self.user_template = user_template
        self.assistant_template = assistant_template
        self._shots_data = shots_data

    @property
    def shots_data(self) -> List[Series]:
        if self._shots_data is None:
            raise ValueError("Shots data not set")
        elif isinstance(self._shots_data, pd.DataFrame):
            return [shot for h, shot in self._shots_data.iterrows()]
        elif isinstance(self._shots_data, list):
            return self._shots_data

    def load_tsv_shots_data(self, filename: str):
        """
        Load information to populate `user` and `assistant` role messages with.
        The column headers should match the placeholders to replace in the prompt templates.
        :param filename: path to the TSV file containing shots data.
        """
        with open(filename, "r") as f:
            self._shots_data = pd.read_csv(f, sep="\t")

    def compose(
        self,
        shots: Iterable = None,
        prompt_input: Union[Dict[str, str], pd.DataFrame, pd.Series] = None,
        system_message: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Core function used to create the list of messages that make up the prompt.
        :param shots: list of hashable values to be used to populate the `user` and `assistant`
                role messages.
                        at the beginning of the prompt, i.e. after the system message and before the shots.
                :param prompt_input: the input for the final `user` role message.
                :param system_message: if `True`, include the `system` role message at the beginning of the prompt.
                :return: list of messages (dictionaries) that make up the prompt.
        """
        prompt = []
        if system_message and self.system_message is not None:
            prompt.append({"role": "system", "content": self.system_message})

        if shots is not None:
            for shot in shots:
                prompt.append(
                    {"role": "user", "content": self.user_template.format_map(shot)}
                )
                prompt.append(
                    {
                        "role": "assistant",
                        "content": self.assistant_template.format_map(shot),
                    }
                )

        if prompt_input is not None:
            if isinstance(prompt_input, pd.DataFrame):
                prompt_input = prompt_input.to_dict("records")[0]

            prompt.append(
                {"role": "user", "content": self.user_template.format_map(prompt_input)}
            )
        return prompt


def load_csv_input_data(filename: str) -> List[Dict[str, str]]:
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def load_schema(filename: str) -> str:
    with open(filename, "r") as f:
        schema = f.read()
        return schema


# def load_schema(filename: str) -> str:
#     with open(filename, "r") as f:
#         return json.load(f)


def load_system_prompt(filename: str) -> str:
    with open(filename, "r") as f:
        system_prompt = f.read()
        return system_prompt


def load_user_prompt(filename: str) -> str:
    with open(filename, "r") as f:
        user_prompt = f.read()
        return user_prompt


# def load_model(model_name: str):
#     return outlines.models.transformers(
#         model_name,
#         model_kwargs={'device_map': 'auto', 'torch_dtype': 'auto'})


def load_model(model_name: str):
    return outlines.models.vllm(
        model_name,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.9,
        enable_prefix_caching="gemma" not in model_name,
        disable_sliding_window="gemma" not in model_name,
        dtype="bfloat16",
        max_model_len=5120,
    )


# def load_model(model_name: str):
#     return LLM(
#         model_name,
#         tensor_parallel_size=torch.cuda.device_count(),
#         gpu_memory_utilization=0.9,
#         enable_prefix_caching=True,
#         disable_sliding_window=True,
#         dtype="bfloat16",
#         max_model_len=4096,
#     )


def export_data(filepath: str, data: list[dict[str, str]]) -> None:
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    data.to_csv(filepath, encoding="utf-8")
    # headers = data[0].keys()
    # with open(filepath, mode="w", encoding="utf-8", newline="") as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=headers)
    #     writer.writeheader()
    #     writer.writerows(data)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main(args, model, tokenizer):
    schema = load_schema(args.schema)
    system_prompt = load_system_prompt(args.system_prompt)
    user_prompt = load_user_prompt(args.user_prompt)

    output_df = None
    if args.output is not None and os.path.exists(args.output):
        logger.info(f"Output file {args.output} already.")
        output_df = pd.read_csv(args.output, index_col="ID")
        # check if the neutrality label as assigned by this model already exists
        if f"neutrality_label_{args.model}" in output_df.columns:
            logger.info(
                f"Neutrality label as assigned by model {args.model} already exists in {args.output}. Skipping..."
            )
            return

    data = load_csv_input_data(args.input)

    sampler = GreedySampler()
    generator = outlines.generate.json(model, schema, sampler=sampler)

    prompt = Prompt(
        user_template=user_prompt,
        assistant_template="{out}",
        system_message=system_prompt,
    )
    prompt.load_tsv_shots_data(args.shots)

    # Iterate over the inputs starting from start_index
    formatted_prompts = list()
    for i, entry in enumerate(data):
        this_prompt = prompt.compose(shots=prompt.shots_data, prompt_input=entry)
        formatted_prompt = tokenizer.apply_chat_template(this_prompt, tokenize=False)
        formatted_prompts.append(formatted_prompt)

    # Pure vLLM
    # guided_decoding_params = GuidedDecodingParams(json=schema, backend="outlines")
    # params = SamplingParams(
    #     temperature=0, max_tokens=1024, guided_decoding=guided_decoding_params
    # )
    # print("Starting the generation")
    # responses = model.generate(formatted_prompts, params)

    # outlines
    responses = generator(formatted_prompts, max_tokens=1024)

    if args.jsonl:
        for i, r in enumerate(responses):
            r["id"] = str(i)
            with open(args.output + ".jsonl", "a") as f:
                json.dump(r, f, ensure_ascii=False)
                f.write("\n")

    labels = [
        "**NEUTRAL**" if r["label"] == "NEUTRAL" else "**GENDERED**" for r in responses
    ]
    # labels = list()
    # for r in responses:
    #     try:
    #         pj = json.loads(r.outputs[0].text)
    #         l = "**NEUTRAL**" if pj["label"] == "NEUTRAL" else "**GENDERED**"
    #     except:
    #         logger.warning(f"Failed to parse response: {r.outputs[0].text}")
    #         l = "**ERROR**"
    #     labels.append(l)

    if output_df is None:
        data_df = pd.DataFrame(data).set_index("ID")
    else:
        data_df = output_df
    data_df[f"neutrality_label_{args.model}"] = labels

    output_path = (
        args.output
        if args.output is not None
        else "out/" + "/".join(args.input.split("/")[-2:])
    )
    export_data(output_path, data_df)
    logger.info(f"Evaluation saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "-i", "--input", type=str, required=True, help="CSV file containing input data."
    # )

    # parser.add_argument(
    #     "-s",
    #     "--system_prompt",
    #     required=True,
    #     help="Text file containing the system message",
    # )
    parser.add_argument(
        "-u",
        "--user_prompt",
        help="Text file containing the user message template",
        default="user",
    )
    # parser.add_argument(
    #     "-d",
    #     "--shots",
    #     help="TSV file containing data for few-shots prompting",
    #     required=True,
    # )
    parser.add_argument(
        "-j", "--schema", help="JSON Schema file", default="schema.json"
    )
    parser.add_argument(
        "-m", "--model", help="Model identifier", default="Qwen/Qwen2.5-72B-Instruct"
    )
    # parser.add_argument("-o", "--output", help="Output file", required=True)
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output."
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Specify to save outputs progressively in JSON Lines format.",
    )

    models_to_evaluate = [
        "meta-llama--Llama-3.1-8B-Instruct",
        "Qwen--Qwen2.5-7B-Instruct",
        "google--gemma-2-9b-it",
        "microsoft--phi-4",
        "meta-llama--Llama-3.3-70B-Instruct",
        "Qwen--Qwen2.5-72B-Instruct",
    ]
    langs = ["es", "de", "el"]
    config = [
        "no_s-no_g-",
        "no_s-",
        "no_g-",
        "",
    ]

    args = parser.parse_args()

    # since it take long, load the model only once
    model = load_model(model_name=args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    for l, m, c in tqdm(
        product(langs, models_to_evaluate, config),
        total=len(langs) * len(models_to_evaluate) * len(config),
    ):
        logger.info(f"Evaluating {m} on {l} with config {c}")
        args.input = f"translation_runs/{l}/data_{m}_prompt_v1-{c}4shot.json.csv"
        args.output = f"{l}/data_{m}_prompt_v1-{c}4shot.json.csv"
        args.system_prompt = f"{l}_prompt"
        args.shots = f"{l}_shots.tsv"
        logging.info(f"Evaluator: {args.model}")
        logging.info(f"Model to evaluate: {m}")
        logging.info(f"Language: {l}")
        logging.info(f"Config: {c}")
        logging.info(f"Input file: {args.input}")
        logging.info(f"Output file: {args.output}")
        logging.info(f"System prompt: {args.system_prompt}")
        logging.info(f"User prompt: {args.user_prompt}")
        logging.info(f"Shots: {args.shots}")
        logging.info(f"Schema: {args.schema}")
        main(args, model, tokenizer)
