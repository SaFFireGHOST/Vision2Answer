# Vision2Answer: Visual Question Answering Pipeline for E-commerce Product Images

## Objectives

* **Data Curation:** Create a VQA dataset with 24,312 QA pairs (19,464 train, 4,848 validation) using the ABO small variant (147,702 listings, 398,212 images).
* **Baseline Evaluation:** Assess pre-trained BLIP, BLIP-2, and ViLT models on the dataset without fine-tuning.
* **Fine-Tuning:** Improve model performance using Low-Rank Adaptation (LoRA) within Kaggle's 2×16GB GPU constraints.
* **Evaluation:** Analyze model performance using Accuracy, F1 Score, BERTScore, and WUP Score, with detailed error analysis by question type.

## Methodology

### 1. Data Curation

* **Merging:** Combined product metadata (listings\_0.json) and image metadata (images.csv) into `cleaned_vqa_metadata_with_images.json` using `merging_final.py`.
* **QA Generation:** Generated 3 diverse QA pairs per image (descriptive, counting, color, function, reasoning) using Gemini 2.0 API with `prompt_final.py`.
* **Dataset:** Produced `vqa_training_data_complete.json` with 24,312 QA pairs across 8,104 images, ensuring single-word answers.

### 2. Model Choices

* **BLIP:** 387M parameters, optimized for VQA (Salesforce/blip-vqa-base).
* **BLIP-2:** 2.7B parameters, robust zero-shot performance (Salesforce/blip2-opt-2.7b).
* **ViLT:** 118M parameters, lightweight VQA model (dandelin/vilt-b32-finetuned-vqa).

### 3. Baseline Evaluation

* **Dataset Split:** 80:20 (19,464 train, 4,848 validation).
* **Metrics:** Accuracy, F1 Score, BERTScore, WUP Score.
* **Results:**

  * BLIP: 52.10% accuracy, strong in color and yes/no questions.
  * BLIP-2: 48.74% accuracy, struggles with counting.
  * ViLT: 36.08% accuracy, weakest in complex queries.

### 4. Fine-Tuning with LoRA

* **Setup:** Rank=16, alpha=16 (BLIP and BLIP-2), 32 (ViLT), targeting attention layers.
* **Optimizations:** KV Cache, mixed precision (FP16 for BLIP-2).
* **Results:**

  * BLIP: 67.35% accuracy (+15.25%), excels in yes/no questions (85.44%).
  * BLIP-2: 54.52% accuracy (+5.78%), improved counting (52.28%).
  * ViLT: 40.99% accuracy (+4.91%), limited gains.

### 5. Evaluation and Error Analysis

* **Metrics:** Quantified performance improvements and question-type-specific gains (e.g., BLIP’s 73.30% in color questions).
* **Error Analysis:** Identified weaknesses in counting (BLIP: 50% error rate) and complex queries (ViLT: 14.43% accuracy in OTHER category).
* **Visualizations:** F1 histograms, question-type boxplots (`metric_distributions.png`).

## Repository Structure

The repository contains the following files:

| File                              | Description                                      |
| --------------------------------- | ------------------------------------------------ |
| `blip_baseline_final.ipynb`       | Baseline evaluation of BLIP model                |
| `blip_lora_final.ipynb`           | Fine-tuning BLIP with LoRA and evaluation        |
| `blip2_baseline_final.ipynb`      | Baseline evaluation of BLIP-2 model              |
| `blip2_lora_final.ipynb`          | Fine-tuning BLIP-2 with LoRA and evaluation      |
| `vilt_baseline_final.ipynb`       | Baseline evaluation of ViLT model                |
| `vilt_lora_final.ipynb`           | Fine-tuning ViLT with LoRA and evaluation        |
| `merging_final.py`                | Merges ABO product and image metadata            |
| `prompt_final.py`                 | Generates QA pairs using Gemini 2.0 API          |
| `conversion.py`                   | Converts JSON dataset to CSV (`vqa_dataset.csv`) |
| `vqa_training_data_complete.json` | VQA dataset with 24,312 QA pairs                 |
| `vqa_training_data_complete.csv`  | CSV version of the dataset                       |
| `Project_Report.pdf`              | Detailed project report                          |

## Installation and Setup

### Prerequisites

* **Hardware:** Kaggle notebook with 2×16GB GPUs (or equivalent)
* **Software:**

  * Python 3.8+
  * Libraries: `torch`, `transformers`, `google-generativeai`, `pandas`, `numpy`, `tqdm`, `requests`, `base64`
  * Gemini 2.0 API key
* **Dataset:** ABO small variant (downloaded separately)

## Usage

### Data Curation

1. Run `merging_final.py` to merge metadata.
2. Run `prompt_final.py` to generate QA pairs.

### Baseline Evaluation

* Execute the notebooks:

  * `blip_baseline_final.ipynb`
  * `blip2_baseline_final.ipynb`
  * `vilt_baseline_final.ipynb`

### Fine-Tuning and Evaluation

* Run the fine-tuning notebooks:

  * `blip_lora_final.ipynb`
  * `blip2_lora_final.ipynb`
  * `vilt_lora_final.ipynb`

### Inference

* Use `inference.py` (included in deliverables) to generate predictions with the fine-tuned BLIP model.

## Results

* **Dataset:** 24,312 QA pairs spanning descriptive, counting, color, function, and reasoning questions.
* **Fine-Tuned Performance:**

  * BLIP (LoRA): 67.35% accuracy, 97.45% BERTScore
  * BLIP-2 (LoRA): 54.52% accuracy, significant counting improvement (+25.35%)
  * ViLT (LoRA): 40.99% accuracy, limited improvements in complex queries
* **Optimizations:** KV Cache, mixed precision (FP16), LoRA reduced memory usage

## Future Work

* **Dataset Expansion:** Increase to 50,000 QA pairs with more counting and functional questions.
* **Model Enhancements:** Explore larger models (e.g., BLIP-2 OPT-6.7B) or ensemble methods.
* **Hyperparameter Tuning:** Conduct broader LoRA searches (e.g., rank=64, varied learning rates).

