"""
This module performs translations using a specified language model and evaluates them.
It supports few-shot prompting with optional system messages and guidelines.
It also computes DA scores for the translations using an automatic metric.
"""

import json
import logging
import os
import pdb
import re
import time

import fire
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from prompts import PromptHelper
from utils import COMETScorer, NeutralScorer, build_prompt_filename, sanitize_model_name

# Set up logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def log_kwargs(**kwargs):
    """
    Log each keyword argument with its name and value.

    Args:
        **kwargs: Arbitrary keyword arguments.
    """
    logger.info("Function arguments:")
    for arg_name, arg_value in kwargs.items():
        logger.info(f"  {arg_name}: {arg_value}")
    return kwargs


def main(
    # dataset: str = None,
    output_dir: str = None,
    model_name_or_path: str = None,
    config_file: str = None,
    config_id: str = None,
    do_eval: bool = False,
    dry_run: bool = False,
    overwrite: bool = False,
    use_system: bool = True,
    use_guidelines: bool = True,
    n_shots: int = 4,
    lang: str = "it",
    DA_model: str = "Unbabel/XCOMET-XL",
):
    gpu_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs: {gpu_count}")

    # 1. parse input params or config file
    if config_file:
        logger.info(f"Using JSON config file:")
        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)

        config_id = str(config_id)
        model_name_or_path = config[config_id]["model"]
        use_system = config[config_id]["use_system"]
        use_guidelines = config[config_id]["use_guidelines"]
        n_shots = config[config_id]["n_shots"]
        lang = config[config_id]["lang"]

    prompt_name = build_prompt_filename(use_system, use_guidelines, n_shots)
    sanitized_model_name = sanitize_model_name(model_name_or_path)
    output_file = os.path.join(
        output_dir, lang, f"data_{sanitized_model_name}_{prompt_name}.csv"
    )
    report_output_file = os.path.join(
        output_dir, lang, f"report_{sanitized_model_name}_{prompt_name}.json"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # don't run if report already exist
    if os.path.exists(report_output_file) and not overwrite:
        logger.info(f"Output file {report_output_file} already exists. Skipping.")
        return

    do_translation = True if not os.path.exists(output_file) or overwrite else False

    log_kwargs(
        config_id=config_id,
        model_name_or_path=model_name_or_path,
        use_system=use_system,
        use_guidelines=use_guidelines,
        n_shots=n_shots,
        lang=lang,
        do_eval=do_eval,
        dry_run=dry_run,
        overwrite=overwrite,
        do_translation=do_translation,
        output_file=output_file,
        report_output_file=report_output_file,
        DA_model=DA_model,
    )

    # 2. load data
    try:
        data = load_dataset("FBK-MT/mGeNTE", f"mGeNTE en-{lang}", split="test")
    except Exception as e:
        logger.info("Exception during dataset loading")
        logger.info(e)
        logger.info("Trying to load the dataset from local files")
        df = pd.read_csv(f"data/en-{lang}.tsv", sep="\t")
        data = Dataset.from_pandas(df)

    if dry_run:
        data = data.select(range(100))

    srcs = data["SRC"]

    # We use two references for COMET evaluation:
    # 1) All references from Europarl's references, which are all gendered
    # 2) References from europarl for Set-G and post-edited/neutralized refs for Set-N
    refs_europarl = data["REF-G"]
    refs_setmatching = [
        data[idx]["REF-G" if data[idx]["SET"] == "Set-G" else "REF-N"]
        for idx in range(len(data))
    ]

    if do_translation:
        # 3. prepare prompt helper to construct prompts accordingly and apply it to srcs
        logger.info(f"Running the translation")
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        system_prompt_as_user = (
            True if model_name_or_path == "google/gemma-2-9b-it" else False
        )
        prompt_helper = PromptHelper(
            use_system,
            use_guidelines,
            n_shots,
            tokenizer,
            lang,
            system_prompt_as_user=system_prompt_as_user,
        )

        texts = prompt_helper.apply_template(srcs)
        logger.info(f"Random text: {texts[np.random.randint(0, len(texts))]}")

        # 4. prepare the model
        guided_decoding_params = GuidedDecodingParams(
            regex=rf"<{lang}>\s\*\*(GENDERED|NEUTRAL)\*\*\s\[[^\]]+\]",
        )
        logger.debug(f"Guided decoding params: {guided_decoding_params}")
        sampling_params = SamplingParams(
            temperature=0,
            max_tokens=256,
            guided_decoding=guided_decoding_params,
        )
        has_rope_scaling = "gemma" in model_name_or_path
        llm = LLM(
            model=model_name_or_path,
            dtype="bfloat16",
            max_model_len=2048,
            tensor_parallel_size=gpu_count,
            gpu_memory_utilization=0.9,
            disable_sliding_window=True if not has_rope_scaling else False,
            enable_prefix_caching=True if not has_rope_scaling else False,
        )

        # 5. generate translations
        outputs = llm.generate(texts, sampling_params)

        # 6. extract translations and labels from the outputs
        translation_labels = list()
        translations = list()
        error_count = 0
        raw_outputs = list()
        for output in outputs:
            text = output.outputs[0].text
            raw_outputs.append(text)
            try:
                translation = re.search(r"\[([^\]]+)\]", text).group(1)
                translations.append(translation)
                translation_labels.append(
                    "**GENDERED**" if "**GENDERED**" in text else "**NEUTRAL**"
                )
            except Exception as e:
                # if the output is badly formatted, let's put a placeholder
                logger.error(f"Error processing output: {text}")
                logger.error(e)
                error_count += 1
                translation_labels.append("**ERROR**")
                translations.append("**ERROR**")

        # store the "translation" and "translation_label" columns in the dataset
        # "translation label" is the label predicted by LM used for the translation
        data = data.add_column("translation_label", translation_labels)
        data = data.add_column("translation", translations)
        data = data.add_column("raw_output", raw_outputs)

        # Classification stats based on the "translation label" on the entire dataset
        y_true = [
            "**GENDERED**" if s == "Set-G" else "**NEUTRAL**" for s in data["SET"]
        ]
        data = data.add_column("set_label", y_true)
        del llm  # free some GPU memory

        # 8. save the translations
        data.to_csv(output_file, index=False, encoding="utf-8")
        logger.info(f"Translations saved to {output_file}")

    else:
        # load translations from the file
        logger.info(f"Loading translations from {output_file}")
        data = pd.read_csv(output_file)
        translations = data["translation"]
        translation_labels = data["translation_label"]
        y_true = data["set_label"]
        error_count = data["translation_label"].value_counts().get("**ERROR**", 0)
        data = Dataset.from_pandas(data)

    # 7. Evaluation

    # 7.1 Classification report on set labels
    valid_labels = ["**GENDERED**", "**NEUTRAL**"]
    class_report = classification_report(
        y_true,
        translation_labels,
        output_dict=True,
        labels=valid_labels,  # use valid_labels to not account for the **ERROR** rows
    )

    # 7.2 Evaluating neutrality with GeNTE's official neutrality detector
    # TODO: this model currently supports en->it only
    class_report_from_classifier_labels = {}
    if lang == "it":
        scorer = NeutralScorer(
            "FBK-MT/GeNTE-evaluator",
            "Musixmatch/umberto-commoncrawl-cased-v1",
            tokenizer_args={"do_lower_case": False},
            device="cuda",
        )

        try:
            neutrality_labels = scorer.predict(translations, batch_size=16)
            neutrality_labels = [
                "**GENDERED**" if n == "gendered" else "**NEUTRAL**"
                for n in neutrality_labels
            ]
        except Exception as e:
            logger.info("Exception during neutrality evaluation")
            logger.info(e)
            neutrality_labels = ["**ERROR**"] * len(translations)

        # save the labels extracted from the official classifier
        if "official_classifier_label" in data.column_names:
            data = data.remove_columns("official_classifier_label")
        data = data.add_column("official_classifier_label", neutrality_labels)
        data.to_csv(output_file, index=False, encoding="utf-8")

        # compute a classification report based on the predicted labels from the official classifier
        class_report_from_classifier_labels = classification_report(
            y_true, neutrality_labels, output_dict=True, labels=valid_labels
        )
        del scorer

    # 7.3 Compute numbers in Set-G separately by GENDER
    pd_data = data.to_pandas()
    for gender in ["F", "M"]:
        curr_data = pd_data.loc[
            (pd_data["SET"] == "Set-G") & (pd_data["GENDER"] == gender)
        ]
        curr_acc = accuracy_score(
            ["**GENDERED**"] * len(curr_data), curr_data["translation_label"]
        )
        class_report["**GENDERED**"][gender] = {
            "accuracy": curr_acc,
            "support": len(curr_data),
        }

    # 7.4 COMET eval on overall quality
    logger.info(f"Evaluating translations with {DA_model}")
    scorer = COMETScorer(DA_model)

    # evaluate on europarl references
    inputs = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(srcs, translations, refs_europarl)
    ]
    gendered_DA_score = scorer.score(inputs, gpus=1).system_score

    # evaluate on setmatching references
    inputs = [
        {"src": src, "mt": mt, "ref": ref}
        for src, mt, ref in zip(srcs, translations, refs_setmatching)
    ]
    setmatching_DA_score = scorer.score(inputs).system_score
    # except Exception as e:
    #     logger.info("Exception during COMET evaluation")
    #     logger.info(e)
    #     gendered_DA_score = None
    #     setmatching_DA_score = None

    # Save all info related to the run
    run_info = {
        "model": model_name_or_path,
        "config_file": config_file,
        "config_id": config_id,
        "extraction_errors": int(error_count),
        "class_report": class_report,
        "official-classifier_report": class_report_from_classifier_labels,
        "gendered_DA_score": gendered_DA_score,
        "setmatching_DA_score": setmatching_DA_score,
        "use_system": use_system,
        "use_guidelines": use_guidelines,
        "n_shots": n_shots,
    }
    with open(report_output_file, "w") as f:
        json.dump(run_info, f, indent=2)

    logger.info(f"Run info saved to {output_dir}")


if __name__ == "__main__":
    stime = time.time()
    fire.Fire(main)
    etime = time.time()
    logger.info(f"Execution time: {etime - stime:.2f} seconds")
    etime = time.time()
    logger.info(f"Execution time: {etime - stime:.2f} seconds")
    logger.info(f"Execution time: {etime - stime:.2f} seconds")
    logger.info(f"Execution time: {etime - stime:.2f} seconds")
