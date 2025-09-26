import argparse
import csv
import json
import logging
import os
import pandas as pd

from openai import OpenAI
from pandas import Series
from typing import Dict, List, Optional, Union, Iterable


def load_csv_input_data(filename: str) -> List[Dict[str, str]]:
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def export_data(filepath: str, data: List[Dict[str, str]]) -> None:
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)

    headers = data[0].keys()
    with open(filepath, mode="w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(data)


class Prompt:
    """
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
    ):
        self.system_message = system_message
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
        :param prompt_input: the input for the final `user` role message.
        :param system_message: if `True`, the `system` role message at the beginning of the prompt.
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


def main(args):
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger = logging.getLogger(__name__)

    with open(args.system, "r") as f:
        system_message = f.read()

    with open(args.user, "r") as f:
        user_message = f.read()

    with open(args.schema, "r") as f:
        schema = json.load(f)

    with open(args.key, "r") as f:
        openai_key = f.read()

    with open(args.org, "r") as f:
        openai_org = f.read()

    client = OpenAI(api_key=openai_key, organization=openai_org)

    prompt = Prompt(
        user_template=user_message,
        assistant_template="{out}",
        system_message=system_message,
    )

    prompt.load_tsv_shots_data(args.shots)

    data = load_csv_input_data(args.input)
    model_outputs = []

    for i, entry in enumerate(data, 1):
        this_prompt = prompt.compose(shots=prompt.shots_data, prompt_input=entry)
        nonempty = False
        temp = 0
        attempt = 1
        while nonempty is False and temp < 1:
            completion = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                temperature=temp,
                messages=this_prompt,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "gnt_eval",
                        "strict": True,
                        "schema": schema,
                    },
                },
            )

            response = completion.choices[0].message.content
            output = json.loads(response)
            if len(output["phrases"]) > 0:
                nonempty = True
            else:
                attempt += 1
                temp += 0.2
                logger.info(
                    f"The model returned an empty phrase list. "
                    f"Increasing temperature to {temp}."
                )

            logger.info(f"{i} ({attempt})\t{output}")

        entry["neutrality_label"] = (
            "**NEUTRAL**" if output["label"] == "NEUTRAL" else "**GENDERED**"
        )
        model_outputs.append(output)

    output_path = (
        args.output
        if args.output is not None
        else ("out/" + "/".join(args.input.split("/")[-2:]))
    )
    export_data(output_path, data)
    logger.info(f"Evaluation saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="CSV file containing input data."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=False,
        default=None,
        help="Path and filename of the output file.",
    )
    parser.add_argument(
        "-s",
        "--system",
        type=str,
        required=True,
        help="Path to the plain text file containing the system message.",
    )
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        required=True,
        help="Path to the plain text file containing the user message.",
    )
    parser.add_argument(
        "-j",
        "--schema",
        type=str,
        required=False,
        default="schema-gpt.json",
        help="Path to the JSON schema file for structured generation.",
    )
    parser.add_argument(
        "-e",
        "--shots",
        type=str,
        required=True,
        help="Path to the TSV file containing the shots.",
    )
    parser.add_argument(
        "-k",
        "--key",
        type=str,
        required=True,
        help="Path to the plain text file containing the OpenAI API key.",
    )
    parser.add_argument(
        "--org",
        type=str,
        required=True,
        help="Path to the plain text file containing the OpenAI organization ID.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output."
    )
    arguments = parser.parse_args()
    main(arguments)
