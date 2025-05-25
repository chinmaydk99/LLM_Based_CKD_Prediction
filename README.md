# Chronic Kidney Disease (CKD) Prediction Project

This project implements machine learning models for predicting Chronic Kidney Disease (CKD) using MIMIC-IV electronic health record data. The project includes data processing pipelines and model training scripts converted from Jupyter notebooks to production-ready Python scripts.

## Project Structure

```
CKD/
├── data_processing/          # Data preprocessing and aggregation scripts
│   ├── prepare_ckd_data.py  # MIMIC-IV data processing and feature extraction
│   └── aggregate_data.py    # Patient data aggregation and text formatting
├── model_training/          # Machine learning model training scripts
│   └── llama_ckd_prediction.py  # Llama-based CKD prediction model
├── utils/                   # Utility functions and helper scripts
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Features

### Data Processing
- **MIMIC-IV Integration**: Processes raw MIMIC-IV data including admissions, diagnoses, lab results, and patient demographics
- **ICD Code Mapping**: Converts ICD-9 and ICD-10 codes to CCSR (Clinical Classifications Software Refined) categories
- **Sequential Patient History**: Creates temporal sequences of patient visits for longitudinal analysis
- **Data Aggregation**: Combines multiple visits per patient into comprehensive medical histories

### Model Training
- **Llama Fine-tuning**: Fine-tunes Llama models using LoRA (Low-Rank Adaptation) for efficient training
- **K-fold Cross-validation**: Implements robust evaluation with stratified k-fold cross-validation
- **Class Imbalance Handling**: Uses balanced class weights to handle imbalanced CKD datasets
- **Comprehensive Metrics**: Evaluates models using accuracy, F1-score, ROC-AUC, and PR-AUC

## Installation

1. Clone the repository and navigate to the CKD directory:
```bash
cd CKD
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation

First, process the raw MIMIC-IV data:

```bash
python data_processing/prepare_ckd_data.py \
    --mimic_dir /path/to/mimic/data \
    --output_dir /path/to/processed/data
```

**Parameters:**
- `--mimic_dir`: Path to directory containing MIMIC-IV CSV files
- `--output_dir`: Directory to save processed train/val/test splits

**Output:**
- `train.pkl`: Training data with patient histories
- `val.pkl`: Validation data
- `test.pkl`: Test data

### 2. Data Aggregation

Aggregate patient data for model training:

```bash
python data_processing/aggregate_data.py \
    --input_dir /path/to/processed/data \
    --output_dir /path/to/aggregated/data
```

**Parameters:**
- `--input_dir`: Directory containing processed pickle files
- `--output_dir`: Directory to save aggregated data

**Output:**
- `aggregated_train_df.pkl`: Aggregated training data
- `aggregated_val_df.pkl`: Aggregated validation data
- `aggregated_test_df.pkl`: Aggregated test data

### 3. Model Training

Train the Llama-based CKD prediction model:

```bash
python model_training/llama_ckd_prediction.py \
    --train_data /path/to/aggregated_train_df.pkl \
    --val_data /path/to/aggregated_val_df.pkl \
    --test_data /path/to/aggregated_test_df.pkl \
    --output_dir /path/to/model/output
```

**Parameters:**
- `--train_data`: Path to aggregated training data
- `--val_data`: Path to aggregated validation data
- `--test_data`: Path to aggregated test data
- `--output_dir`: Directory to save trained models and results
- `--model_id`: Hugging Face model identifier (default: "unsloth/Llama-3.2-1B-bnb-4bit")
- `--k_folds`: Number of folds for cross-validation (default: 3)
- `--seed`: Random seed for reproducibility (default: 42)

**Output:**
- `best_model/`: Directory containing the best trained model
- `training_results.txt`: Detailed training and evaluation results

## Data Pipeline

### 1. Raw Data Processing (`prepare_ckd_data.py`)

**Intuition**: This script transforms raw MIMIC-IV data into a structured format suitable for machine learning. It creates sequential patient histories that capture the temporal progression of medical conditions.

**Key Steps:**
1. **Data Loading**: Loads MIMIC-IV tables (admissions, diagnoses, lab events, etc.)
2. **ICD Mapping**: Converts diagnostic codes to standardized CCSR categories
3. **Lab Processing**: Identifies abnormal lab values relevant to kidney function
4. **Patient Filtering**: Selects patients with multiple visits and no initial CKD diagnosis
5. **Sequence Creation**: Builds cumulative medical histories for each patient
6. **Train/Val/Test Split**: Creates balanced splits while maintaining patient-level separation

### 2. Data Aggregation (`aggregate_data.py`)

**Intuition**: This script aggregates multiple visits per patient into a single comprehensive record, creating a holistic view of each patient's medical history for classification.

**Key Steps:**
1. **Data Flattening**: Converts nested patient histories into flat records
2. **Feature Aggregation**: Combines diagnoses and lab results across all visits
3. **Text Formatting**: Creates natural language descriptions of patient histories
4. **Label Assignment**: Determines CKD status based on any positive diagnosis

### 3. Model Training (`llama_ckd_prediction.py`)

**Intuition**: This script fine-tunes a large language model to understand medical text and predict CKD risk. It uses LoRA for efficient training and k-fold validation for robust evaluation.

**Key Steps:**
1. **Text Preprocessing**: Formats patient data into natural language descriptions
2. **Model Setup**: Configures Llama model with LoRA adapters for efficient fine-tuning
3. **Cross-Validation**: Trains multiple models using k-fold CV for robust evaluation
4. **Evaluation**: Computes comprehensive metrics including ROC-AUC and PR-AUC
5. **Model Selection**: Saves the best performing model based on validation metrics

## Key Concepts

### CCSR Mapping
Clinical Classifications Software Refined (CCSR) provides a standardized way to group ICD codes into clinically meaningful categories. This reduces the dimensionality of diagnostic codes while preserving clinical relevance.

### LoRA Fine-tuning
Low-Rank Adaptation (LoRA) allows efficient fine-tuning of large language models by training only a small number of additional parameters, making it feasible to adapt models like Llama for medical tasks.

### Sequential Patient Modeling
The project models patients as sequences of medical encounters, capturing the temporal evolution of health conditions. This is crucial for predicting future diagnoses like CKD.

## Performance Metrics

The models are evaluated using multiple metrics appropriate for medical prediction tasks:

- **Accuracy**: Overall classification accuracy
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **PR-AUC**: Area under the precision-recall curve (important for imbalanced data)

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for model training)
- MIMIC-IV dataset access
- Sufficient storage for processed data (~several GB)

## Notes

- The project is designed to work with MIMIC-IV data, which requires credentialed access
- Model training can be computationally intensive and benefits from GPU acceleration
- The scripts include comprehensive logging for monitoring progress and debugging
- All random seeds are set for reproducible results

## Contributing

When adding new features or models:
1. Follow the existing code structure and documentation patterns
2. Include comprehensive docstrings and type hints
3. Add appropriate logging and error handling
4. Update this README with new functionality 