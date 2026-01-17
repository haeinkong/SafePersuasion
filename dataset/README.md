## Overview

This dataset consists of human-written persuasive comments annotated for **persuasion safety** — specifically distinguishing between *rational persuasion* and *manipulation*.  

## Datasets

There are three datasets used in our experiments. Details are below.

| File | Description |
|--------|--------------|
| `SafePersuasion.csv` | The entire dataset. It was used for the binary prediction (first-level label detection). |
| `SafePersuasion_Multi_M.csv` | The dataset used for the multi-label prediction of manipulation (second-level label detection). |
| `SafePersuasion_Multi_RP.csv` | The dataset used for the multi-label prediction of rational persuasion (second-level label detection). |

This dataset includes the persuasive text, its corresponding first and second-level label(s). **Please read our papers for details such as definitions and types of labels.**

| Field | Description |
|--------|--------------|
| `text` | The persuasive message or comment text |
| `first_label` | The first-level label, either rational persuasion or manpulation |
| `second_label` | The second-level label, such as `Logical Appeal`, `Negative Emotional Appeal`, etc. |
---
