import time
import fire
import pandas as pd
import json
from utils import NeutralScorer, sanitize_model_name, build_prompt_filename
import logging
from tqdm import tqdm
import os


logger = logging.getLogger(__name__)


def main(config_file: str, input_dir: str):
    with open(config_file) as f:
        config = json.load(f)
    
    scorer = NeutralScorer(
        "FBK-MT/GeNTE-evaluator",
        "Musixmatch/umberto-commoncrawl-cased-v1",
        tokenizer_args={"do_lower_case": False},
        device="cuda",
    )
    logger.info("Loaded NeutralScorer")

    config = {k: v for k, v in config.items() if v["lang"] == "it"}
    logger.info(f"Loaded {len(config)} configurations for Italian")

    for idx, values in tqdm(config.items()):
        sanitized_model_name = sanitize_model_name(values["model"])
        prompt_name = build_prompt_filename(values["use_system"], values["use_guidelines"], values["n_shots"])
        file_with_translations = os.path.join(
            input_dir, values["lang"],
            f"data_{sanitized_model_name}_{prompt_name}.csv"
        )
        df = pd.read_csv(file_with_translations)
        translations = df["translation"].tolist()
        neutrality_labels = scorer.predict(translations, batch_size=16)
        neutrality_labels = [
            "**GENDERED**" if n == "gendered" else "**NEUTRAL**"
            for n in neutrality_labels
        ]
        df["neutrality_label"] = neutrality_labels
        df.to_csv(file_with_translations, index=False)


if __name__ == '__main__':
    stime = time.time()
    fire.Fire(main)
    etime = time.time()
    logger.info(f"Execution time: {etime - stime:.2f} seconds")