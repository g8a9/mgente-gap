<!-- [![View of Trento](banner.jpg)](https://mt.fbk.eu/) -->

Code associated with the paper: **Mind the Inclusivity Gap: Multilingual Gender-Neutral Translation Evaluation with mGeNTE**. 

[![HuggingFace Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://huggingface.co/datasets/FBK-MT/mGeNTE)
[![arxiv](https://img.shields.io/badge/arxiv-paper-red)](https://arxiv.org/abs/2501.09409)

## Getting Started

To replicate our experiments, we recommend working in isolation in a new python environment. Once a new environment is created, run
```bash
pip install -r requirements.txt
```

## Codebase Organization and Use

The codebase will let you run the four main experimental components of the paper. Each script will require minimal changes to adapt to your setup, e.g., correct input/output directories, etc.

> [!IMPORTANT]
> We used a SLURM-based HPC to run our experiments. Some bash script and organization require you to be in the same situation or minimal changes to be run on a standard workstation. If anything is not clear, please open an issue on this repository.

### 1. Translating mGeNTE

Use the script `bash/translate_runs.sh` to translate mGeNTE across all models, languages, and using the correct configurations. Input parameters for each run are in the file `config/translate_runs.sh`. This script's logic is based on running one translation run per SLURM job using arrayjobs.

### 2. Gender Neutral Evaluation of Translations

Once translations are generated, you can assing a neutrality label using the code in `src/gnt_eval`. Please refer to the README.md in that folder for details. 

### 3. Computing Attributions using AttnLRP

Use the script `bash/attribute_attnlrp.sh` to compute fine-grained token attributions. 

> [!TIP]
> We are in the process of releasing **all** the attributions we computed so you won't have to (provided that you are interested in attributing the same models).

### 4. Analyzing the Attribution Scores

Use the Jupyter Notebook `notebooks/analize_attnlrp.ipynb` to analyze, aggregate, and postprocess the raw attribute scores computed in the previous step. You may want to run this script to compute an intermediate representation with statistics of which part of the context was most used for which translation example.

## Citation

If you use any of the materials related to the paper, please cite:

```bibtex
@misc{savoldi2025mindinclusivitygapmultilingual,
      title={Mind the Inclusivity Gap: Multilingual Gender-Neutral Translation Evaluation with mGeNTE}, 
      author={Beatrice Savoldi and Giuseppe Attanasio and Eleonora Cupin and Eleni Gkovedarou and Janiça Hackenbuchner and Anne Lauscher and Matteo Negri and Andrea Piergentili and Manjinder Thind and Luisa Bentivogli},
      year={2025},
      eprint={2501.09409},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.09409}, 
}
```