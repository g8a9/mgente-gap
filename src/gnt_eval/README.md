# GPT-based evaluation

To run GNT evaluation with GPT-4o use the script `gpt4_evaluate.py` with the following arguments:

## Running the Script
Use the following command structure to execute the script:

```
python gpt_evaluate.py -i <input.csv> -s <system_prompt.txt> -u <user_prompt_format.txt> -j <schema.json> -e <shots.tsv> -k <openai_api_key.txt> --org <openai_org_id.txt> [optional arguments]
```

Required Arguments:
* -i, --input: CSV file containing input data

* -s, --system: Plain text file containing the system message

* -u, --user: Plain text file containing the user message with placeholders for the shot and input data

* -e, --shots: TSV file containing evaluation examples (shots)

* -k, --key: File containing your OpenAI API key

* --org: File containing your OpenAI organization ID


Optional Arguments:

-o, --output: Path and filename for the output file \
Default: None (prints output to terminal) \
Example: -o results/output.json

-m, --model: OpenAI model ID (must support structured generation) \
Default: gpt-4o-2024-08-06 \
Example: -m gpt-4o-2024-11-20

-v, --verbose: Enable verbose output for debugging purposes \
Default: Disabled \
Example: -v

-j, --schema: JSON schema file for structured generation \
Default: schema-gpt.json


# Local LLM-based evaluation

Make sure to install the requirements listed in `requirements.txt`.

Use the following command structure to execute the script:

```
python local_evaluate.py -i <input.csv> -s <system_prompt.txt> -u <user_prompt_format.txt> -j <schema.json> -e <shots.tsv> [optional arguments]
```

Required Arguments:
* -i, --input: CSV file containing input data

* -s, --system: Plain text file containing the system message

* -e, --shots: TSV file containing evaluation examples (shots)

* -o, --output: Path and filename for the output file

Optional Arguments:


Default: None (prints output to terminal)
Example: -o results/output.json

-m, --model: Hugging Face model identifier. The model will be downloaded if it is not found in the cache. \
Default: Qwen/Qwen2.5-72B-Instruct \
Example: -m mistralai/Mistral-Small-24B-Instruct-2501

-u, --user: Plain text file containing the user message with placeholders for the shot and input data \
Default: user

-v, --verbose: Enable verbose output for debugging purposes \
Default: Disabled \
Example: -v

-j, --schema: JSON schema file for structured generation \
Default: schema.json

--jsonl: Specify to save model outputs progressively in a JSON-Lines file

# Run the official GeNTE classifier 

We provide a utility script to run the official neutrality classifier from [GeNTE](https://aclanthology.org/2023.emnlp-main.873/). 
Note, the classifier should only be used to evaluate **Italian** translations. If you are using it, please cite:

```bibtex
@inproceedings{piergentili-etal-2023-hi,
    title = "Hi Guys or Hi Folks? Benchmarking Gender-Neutral Machine Translation with the {G}e{NTE} Corpus",
    author = "Piergentili, Andrea  and
      Savoldi, Beatrice  and
      Fucci, Dennis  and
      Negri, Matteo  and
      Bentivogli, Luisa",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.873/",
    doi = "10.18653/v1/2023.emnlp-main.873",
    pages = "14124--14140",
    abstract = "Gender inequality is embedded in our communication practices and perpetuated in translation technologies. This becomes particularly apparent when translating into grammatical gender languages, where machine translation (MT) often defaults to masculine and stereotypical representations by making undue binary gender assumptions. Our work addresses the rising demand for inclusive language by focusing head-on on gender-neutral translation from English to Italian. We start from the essentials: proposing a dedicated benchmark and exploring automated evaluation methods. First, we introduce GeNTE, a natural, bilingual test set for gender-neutral translation, whose creation was informed by a survey on the perception and use of neutral language. Based on GeNTE, we then overview existing reference-based evaluation approaches, highlight their limits, and propose a reference-free method more suitable to assess gender-neutral translation."
}
```