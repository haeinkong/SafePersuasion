# SafePersuasion: A Dataset, Taxonomy, and Baselines for Analysis of Rational Persuasion and Manipulation (IJCNLP/AACL Findings 2025)

This is a repository for **[SafePersuasion](https://aclanthology.org/2025.findings-ijcnlp.65/)**

## Abstract

Persuasion is a central feature of communication, widely used to influence beliefs, attitudes, and behaviors. In today’s digital landscape, across social media and online platforms, persuasive content is pervasive, appearing in political campaigns, marketing, fundraising appeals, and more. These strategies span a broad spectrum, from rational and ethical appeals to highly manipulative tactics, some of which pose significant risks to individuals and society. Despite the growing need to identify and differentiate safe from unsafe persuasion, empirical research in this area remains limited. To address this gap, we introduce `SafePersuasion`, a two-level taxonomy and annotated dataset that categorizes persuasive techniques based on their safety. We evaluate the baseline performance of three large language models in detecting manipulation and its subtypes, and report only moderate success in distinguishing manipulative content from rational persuasion. By releasing `SafePersuasion`, we aim to advance research on detecting unsafe persuasion and support the development of tools that promote ethical standards and transparency in persuasive communication online

## Project Structure

```
── dataset
│   ├── README.md # Dataset Description
│   ├── SafePersuasion.csv
│   ├── SafePersuasion_Multi_M.csv
│   └── SafePersuasion_Multi_RP.csv
└── src
    ├── loader.py # Data Loading
    ├── main.py # Main experiment code
    ├── models/ # Model utilities
    └── prompts/ # Prompt templates
├── LICENSE
├── README.md
├── requirements.txt
```

## Installation
```bash
git clone https://github.com/haeinkong/SafePersuasion.git
cd SafePersuasion
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py --model [MODEL] --task [TASK] --prompt [PROMPT]

# Example
python src/main.py --model gpt --task binary --prompt zero-shot
```

**Options:**
- `--model`: `gpt`, `claude`, `llama`
- `--task`: `binary`, `multi-m`, `multi-rp`  
- `--prompt`: `zero-shot`, `few-shot`, `chain-of-thought`


## License
- **Code:** MIT License
- **Dataset:** CC BY-NC 4.0 (Non-commercial use only)

See [LICENSE](LICENSE) for details.

## Citation 
If you find this work useful, please cite:
```bibtex
@inproceedings{kong2025safepersuasion,
  title={SafePersuasion: A Dataset, Taxonomy, and Baselines for Analysis of Rational Persuasion and Manipulation},
  author={Haein Kong, A M Muntasir Rahman, Ruixiang Tang, and Vivek Singh},
  booktitle={Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics},
  pages = {1097--1111}
  year={2025},
  url={https://aclanthology.org/2025.findings-ijcnlp.65/}
}
```

