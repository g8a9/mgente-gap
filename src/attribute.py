"""
This module computes token-level contributions of pre-computed translations.
We use Attention LRP as it is implemented in the LXT library.
"""

import tyro
import logging
from tqdm import tqdm

import pandas as pd
from prompts import PromptHelper
from xai import AttLRPHelper, AttributionOutput, RowAttribution, AttributionUnit
import pickle
import os

logging.basicConfig(
    format="%(asctime)s,%(levelname)s: %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)


def find_spans(l: list, sl: list):
    """Find the first span that matches the sublist sl in list l."""
    assert len(l) >= len(sl)
    for i in range(len(l) - len(sl) + 1):
        if l[i : i + len(sl)] == sl:
            return i, i + len(sl)


def process_file(
    file_path,
    attributor,
    prompt_helper,
):
    df = pd.read_csv(
        file_path, sep="\t" if file_path.endswith(".tsv") else ",", index_col="ID"
    )
    logger.debug(f"Processing file: {file_path}")
    logger.debug(f"File shape: {df.shape}")

    # remove rows that have "nan-empty" in the gold_neutrality_label column
    if "gold_neutrality_label" in df.columns:
        df = df.loc[df["gold_neutrality_label"] != "nan-empty"]

    attribution_output = AttributionOutput(
        file_path=file_path,
        model_name=attributor.model_name_or_path,
    )

    for idx, row in tqdm(df.iterrows(), desc="Rows", total=len(df)):
        row_attribution = RowAttribution(rid=idx)

        # the entire model input / prompt
        input_text = prompt_helper.apply_template(row.SRC)[0]
        tokenized_input = prompt_helper.tokenize(input_text, return_ids=True)
        input_tokens, input_ids = tokenized_input[0][0], tokenized_input[1][0]
        row_attribution.full_prompt = AttributionUnit(
            tokens=input_tokens,
            span=(0, len(input_tokens)),
        )

        # source to translate
        text = prompt_helper.tokenize(f"<en> {row.SRC}")[0]
        span = find_spans(input_tokens, text)
        assert span is not None, f"span at {idx} was not found. {input_tokens} {text}"
        row_attribution.source = AttributionUnit(
            tokens=text,
            span=span,
        )

        # generated output
        forced_target = row.raw_output
        tokenized_output = prompt_helper.tokenize(forced_target, return_ids=True)
        output_tokens, output_ids = tokenized_output[0][0], tokenized_output[1][0]
        row_attribution.target = AttributionUnit(
            tokens=output_tokens,
            span=(0, len(output_tokens)),
        )

        # tokens of the translation label. Note it is following a whitespace
        tl_tokens = prompt_helper.tokenize(" " + row.translation_label)[0]
        # tl span is relative to the output tokens
        tl_span = find_spans(output_tokens, tl_tokens)
        assert tl_span is not None
        row_attribution.translation_label = AttributionUnit(
            tokens=tl_tokens,
            span=tl_span,
            # attributed tokens: the entire input string
            # attributing tokens: the translation label
            metadata={
                "attributions": attributor.get_attributions(
                    input_ids, output_ids, (0, len(input_tokens)), tl_span
                )
            },
        )

        # keeping the square brackes as models can predict the last token as ".]"
        tr_tokens = prompt_helper.tokenize(f" [{row.translation}]")[0]
        tr_span = find_spans(output_tokens, tr_tokens)
        assert tr_span is not None
        row_attribution.translation = AttributionUnit(
            tokens=tr_tokens,
            span=tr_span,
            # attributed tokens: the entire input string plus <lang> and the translation label
            # attributing tokens: the translation
            metadata={
                "attributions": attributor.get_attributions(
                    input_ids,
                    output_ids,
                    (0, len(input_tokens) + tl_span[1]),
                    tr_span,
                )
            },
        )

        attribution_output.rows.append(row_attribution)

    # system prompt
    tokens = prompt_helper.get_part_tokens("system_prompt")[0]
    attribution_output.system_prompt = AttributionUnit(
        tokens=tokens,
        span=find_spans(input_tokens, tokens),
    )
    # guidelines
    tokens = prompt_helper.get_part_tokens("gender_neutral_guidelines")[0]
    attribution_output.guidelines = AttributionUnit(
        tokens=tokens,
        span=find_spans(input_tokens, tokens),
    )
    # preamble
    tokens = prompt_helper.get_part_tokens("preamble")[0]
    attribution_output.preamble = AttributionUnit(
        tokens=tokens,
        span=find_spans(input_tokens, tokens),
    )

    # demonstrations
    demo_units = list()
    for (u_t, a_t), st in zip(
        prompt_helper.get_part_tokens("demonstrations"),
        prompt_helper.shots_type,
    ):
        u_t_span = find_spans(input_tokens, u_t[0])
        a_t_span = find_spans(input_tokens, a_t[0])
        demo_units.append(
            (
                AttributionUnit(tokens=u_t[0], span=u_t_span, metadata={"type": st}),
                AttributionUnit(tokens=a_t[0], span=a_t_span, metadata={"type": st}),
            )
        )

    attribution_output.demonstrations = demo_units
    return attribution_output


def main(
    input_files: str | list[str],
    output_dir: str,
    model_name: str,
    n_shots: int = 4,
    use_system: bool = True,
    use_guidelines: bool = True,
):
    logger.info("Starting token-level contribution computation.")
    logger.info(f"Files: {input_files}")
    if not isinstance(input_files, list):
        input_files = [input_files]

    # extract the target_lang from the filename (e.g., if _es_ or _it_ are in it)
    target_langs = list()
    for file in input_files:
        if "_es_" in file:
            target_langs.append("es")
        elif "_it_" in file:
            target_langs.append("it")
        elif "_de_" in file:
            target_langs.append("de")
        elif "_el_" in file:
            target_langs.append("el")
        else:
            raise ValueError("Target lang not found")

    logger.info(f"Parsed target langs:\n{target_langs}")

    # this class will handle for us model loading, tokenization, and other utils
    attributor = AttLRPHelper(model_name)
    tokenizer = attributor.tokenizer
    system_prompt_as_user = True if model_name == "google/gemma-2-9b-it" else False

    for file, tl in tqdm(zip(input_files, target_langs), desc="Processing files"):
        ph = PromptHelper(
            use_system=use_system,
            use_guidelines=use_guidelines,
            n_shots=n_shots,
            lang=tl,
            tokenizer=tokenizer,
            system_prompt_as_user=system_prompt_as_user,
        )

        out = process_file(file, attributor, ph)
        basename = os.path.basename(file)
        with open(
            os.path.join(output_dir, os.path.splitext(basename)[0]) + ".attr.pkl", "wb"
        ) as fp:
            pickle.dump(out, fp)


if __name__ == "__main__":
    tyro.cli(main)
