<!-- [![View of Trento](banner.jpg)](https://mt.fbk.eu/) -->

Code associated with the paper: **Mind the Inclusivity Gap: Multilingual Gender-Neutral Translation Evaluation with mGeNTE**. 

[![HuggingFace Dataset](https://img.shields.io/badge/HF-Dataset-yellow)](https://huggingface.co/datasets/FBK-MT/mGeNTE)
[![arxiv](https://img.shields.io/badge/arxiv-paper-red)](https://arxiv.org/abs/2501.09409)

## Getting Started

To replicate our experiments, we recommend working in isolation in a new python environment. Once a new environment is created, run
```bash
pip install -f requirements.txt
```

## Codebase Organization and Use

The codebase will let you run the four main experimental components of the paper.

> [!IMPORTANT]
> We used a SLURM-based HPC to run our experiments. Some bash script and organization require you to be in the same situation or minimal changes to be run on a standard workstation. If anything is not clear, please open an issue on this repository.

### 1. Translating mGeNTE

### 2. Gender Neutral Evaluation of Translations

### 3. Computing Attributions using AttnLRP

### 4. (Jupyter) Analyzing the Attribution Scores


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