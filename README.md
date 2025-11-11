# _You Are What You Train_: Effects of Data Composition on Training Context-aware Machine Translation Models

This repository contains the code for the paper "You Are What You Train: Effects of Data Composition on Training Context-aware Machine Translation Models" ([https://arxiv.org/abs/2509.14031](https://arxiv.org/abs/2509.14031)).

## Requirements
This repository was created in Python `3.12`. We use huggingface datasets framework. 
The dependencies are listed in `requirements.txt`. You can install them using pip:

```bash
pip install -r requirements.txt
```

## Data
In this paper, we utilized the following datasets:
- **IWSLT**: This dataset is downloaded automatically from huggingface.
- **ctxPro**: Clone and setup this repository from [https://github.com/rewicks/ctxpro](https://github.com/rewicks/ctxpro). After setting up, copy the file `data/crtxPro/lang_pair_sets.yaml` from this repository to the `ctxPro_path/release/`
- **OpenSubtitles**: We use the ctxPro to download OpenSubtitles dataset. Point the `raw_dataset_path` field in the configuration files used to run experiments to the directory `ctxPro_path/data/opensubs/` (where the OpenSubtitles dataset is downloaded).
- **ctxPro based on IWSLT**: We annotated IWSLT dataset using ctxPro toolset. The annotations are available in the `data/iwslt2017_ctxpro/en-de/ctxpro` folder.

## Training
The configuration of the training of the models is done through `yaml` configuration files. The paths in those files have to be adjusted to point to the datasets.

The example configuration files are provided in the `experiments` folder. The training of the models is done through the `training/train_ctx_aware_opus_mt.py` script. The script should be run from `experiments/opus_mt/baseline_seed-` folder:
```bash
PYTHONPATH=path/to/repo/src/ python -m training.train_ctx_aware_opus_mt --configs training_config.yaml
```

## Evaluation
The evaluation is configured by a series of `yaml` files.

- Translate IWSLT:
```bash
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_translate --configs base_config.yaml iwslt_translate_config.yaml
```
- Translate OpenSubtitles:
```bash
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_translate --configs base_config.yaml os_translate_config.yaml
```
- Evaluate ctxPro:
```bash
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_ctxpro --configs base_config.yaml ctxpro_config.yaml
```
- Contrapro:
```bash
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_contrapro --configs base_config.yaml contrapro_config.yaml
```

The evaluation in the multilingual setting is slightly different due to the number of language directions. Navigate to the folder representing the language pair you want to evaluate. For example, for the `en-de` language pair navigate to `model/ende/` and run the evaluation scripts (note that `base_config.yaml` is in the parent folder):
- Translate IWSLT:
```bash
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_translate --configs ../base_config.yaml iwslt_translate_config.yaml
```
- Translate OpenSubtitles:
```bash
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_translate --configs ../base_config.yaml ../base_os_translate_config.yaml os_translate_config.yaml
```
- Evaluate ctxPro:
```bash
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_ctxpro --configs ../base_config.yaml ../base_ctxpro_config.yaml ctxpro_config.yaml
```
- Contrapro:
```bash
PYTHONPATH=path/to/repo/src/ python -m evaluating.opus_mt_contrapro --configs ../base_config.yaml contrapro_config.yaml
```

LLM-based experiments are evaluated in a similar way.

## AI Assistant Usage

During writing the code in this repository we used Github Copilot (https://github.com/features/copilot).


## Citation

Please use the following citation if you find this repository or the accompanying paper useful for your research:

```
@inproceedings{maka-etal-2025-train,
    title = "You Are What You Train: Effects of Data Composition on Training Context-aware Machine Translation Models",
    author = "Maka, Pawel  and
      Semerci, Yusuf Can  and
      Scholtes, Jan  and
      Spanakis, Gerasimos",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1394/",
    doi = "10.18653/v1/2025.emnlp-main.1394",
    pages = "27402--27425",
    ISBN = "979-8-89176-332-6",
    abstract = "Achieving human-level translations requires leveraging context to ensure coherence and handle complex phenomena like pronoun disambiguation. Sparsity of contextually rich examples in the standard training data has been hypothesized as the reason for the difficulty of context utilization. In this work, we systematically validate this claim in both single- and multilingual settings by constructing training datasets with a controlled proportions of contextually relevant examples. We demonstrate a strong association between training data sparsity and model performance confirming sparsity as a key bottleneck. Importantly, we reveal that improvements in one contextual phenomenon do no generalize to others. While we observe some cross-lingual transfer, it is not significantly higher between languages within the same sub-family. Finally, we propose and empirically evaluate two training strategies designed to leverage the available data. These strategies improve context utilization, resulting in accuracy gains of up to 6 and 8 percentage points on the ctxPro evaluation in single- and multilingual settings respectively."
}
```