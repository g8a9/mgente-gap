from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import torch
from datasets import Dataset
from tqdm import tqdm
import logging
from comet import download_model, load_from_checkpoint
import matplotlib.cm as cm
import matplotlib.colors as colors
import os
import subprocess
from pathlib import Path
import re


logger = logging.getLogger(__name__)


inference_decorator = (
    torch.inference_mode if torch.__version__ >= "2.0.0" else torch.no_grad
)


def build_prompt_filename(use_system, use_guidelines, n_shots):
    components = list()
    if not use_system:
        components.append("no_s")
    if not use_guidelines:
        components.append("no_g")
    components.append(f"{n_shots}shot")
    return f"prompt_v1-{'-'.join(components)}.json"


def sanitize_model_name(model_name):
    return model_name.replace("/", "--")


def log_arguments(logger):
    def decorator(func):
        def wrapper(*args, **kwargs):
            arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
            args_dict = {arg_name: arg for arg_name, arg in zip(arg_names, args)}
            args_dict.update(kwargs)
            logger.info(f"Arguments for {func.__name__}: {args_dict}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


class NeutralScorer:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: str,
        tokenizer_args: dict = {},
        device="cuda",
        torch_dtype=torch.bfloat16,
    ):
        self.device = device

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, torch_dtype=torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, **tokenizer_args)
        if not hasattr(self.tokenizer, "pad_token_id"):
            logger.info("Setting pad token to eos token")
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.model.to(device).eval()

    @inference_decorator()
    def predict(self, texts, batch_size=4, num_workers=0):
        data = Dataset.from_dict({"text": texts})
        data = data.map(
            lambda x: self.tokenizer(x["text"], truncation=True),
            batched=True,
            remove_columns=["text"],
        )
        collator = DataCollatorWithPadding(
            self.tokenizer, pad_to_multiple_of=8, return_tensors="pt"
        )
        loader = torch.utils.data.DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

        final_preds = list()
        for step, batch in tqdm(
            enumerate(loader), desc="Batch", total=len(texts) // batch_size
        ):
            batch.to(self.device)
            outputs = self.model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            predictions = [self.model.config.id2label[i.item()] for i in predictions]
            final_preds.extend(predictions)

        return final_preds


class COMETScorer:
    def __init__(self, model_name_or_path: str) -> None:
        path = download_model(model_name_or_path)
        self.model = load_from_checkpoint(path)

    def score(self, texts, batch_size=2, gpus=torch.cuda.device_count()):
        return self.model.predict(texts, batch_size=batch_size, gpus=gpus)


def _apply_colormap(relevance, cmap):

    colormap = cm.get_cmap(cmap)
    return colormap(colors.Normalize(vmin=-1, vmax=1)(relevance))


def _generate_latex(words, relevances, cmap="bwr", append_string: str = None):
    """
    Generate LaTeX code for a sentence with colored words based on their relevances.
    """

    # Generate LaTeX code
    latex_code = r"""
    \documentclass[varwidth=200mm]{standalone}
    \usepackage[dvipsnames]{xcolor}
    \usepackage[utf8]{inputenc}
    \begin{document}
    \fbox{
    \parbox{0.7\textwidth}{
    \setlength\fboxsep{0pt}
    """

    for word, relevance in zip(words, relevances):
        rgb = _apply_colormap(relevance, cmap)
        r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)

        if word.startswith(" "):
            latex_code += f" \\colorbox[RGB]{{{r},{g},{b}}}{{\\strut {word}}}"
        else:
            latex_code += f"\\colorbox[RGB]{{{r},{g},{b}}}{{\\strut {word}}}"

    if append_string:
        latex_code += append_string

    latex_code += r"}}\end{document}"

    return latex_code


def _compile_latex_to_pdf(
    latex_code, path="word_colors.pdf", delete_aux_files=True, backend="xelatex"
):
    """
    Compile LaTeX code to a PDF file using pdflatex or xelatex.
    """

    # Save LaTeX code to a file
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)

    with open(path.with_suffix(".tex"), "w") as f:
        f.write(latex_code)

    # Use pdflatex to generate PDF file
    if backend == "pdflatex":
        subprocess.call(
            ["pdflatex", "--output-directory", path.parent, path.with_suffix(".tex")]
        )
    elif backend == "xelatex":
        subprocess.call(
            ["xelatex", "--output-directory", path.parent, path.with_suffix(".tex")]
        )

    print("PDF file generated successfully.")

    if delete_aux_files:
        for suffix in [".aux", ".log", ".tex"]:
            os.remove(path.with_suffix(suffix))


def pdf_heatmap(
    words,
    relevances,
    cmap="bwr",
    path="heatmap.pdf",
    delete_aux_files=True,
    backend="xelatex",
    append_string: str = None,
):
    """
    Generate a PDF file with a heatmap of the relevances of the words in a sentence using LaTeX.

    Parameters
    ----------
    words : list of str
        The words in the sentence.
    relevances : list of float
        The relevances of the words normalized between -1 and 1.
    cmap : str
        The name of the colormap to use.
    path : str
        The path to save the PDF file.
    delete_aux_files : bool
        Whether to delete the auxiliary files generated by LaTeX.
    backend : str
        The LaTeX backend to use (pdflatex or xelatex).
    """

    assert len(words) == len(
        relevances
    ), "The number of words and relevances must be the same."
    assert (
        relevances.min() >= -1 and relevances.max() <= 1
    ), "The relevances must be normalized between -1 and 1."

    latex_code = _generate_latex(
        words, relevances, cmap=cmap, append_string=append_string
    )

    _compile_latex_to_pdf(
        latex_code, path=path, delete_aux_files=delete_aux_files, backend=backend
    )


def clean_tokens(words):
    """
    Clean wordpiece tokens by removing special characters and splitting them into words.
    """

    if any("▁" in word for word in words):
        words = [word.replace("▁", " ") for word in words]

    elif any("Ġ" in word for word in words):
        words = [word.replace("Ġ", " ") for word in words]

    elif any("##" in word for word in words):
        words = [
            word.replace("##", "") if "##" in word else " " + word for word in words
        ]
        words[0] = words[0].strip()

    else:
        raise ValueError("The tokenization scheme is not recognized.")

    special_characters = ["&", "%", "$", "#", "_", "{", "}", "\\"]
    for i, word in enumerate(words):
        for special_character in special_characters:
            if special_character in word:
                words[i] = word.replace(special_character, "\\" + special_character)

    # words = [w.replace("Ċ", r" \\ ") for w in words]
    return words
