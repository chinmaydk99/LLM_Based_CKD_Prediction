#!/usr/bin/env python3
"""
Data Aggregation Script for CKD Prediction

This script aggregates patient data from the processed MIMIC-IV dataset,
combining multiple visits per patient into a single record with comprehensive
medical history for CKD prediction models.

Usage:
    python aggregate_data.py --input_dir /path/to/processed/data --output_dir /path/to/output
"""

import pandas as pd
import argparse
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def flatten_patient_data(data_list):
    """
    Flatten nested patient data structure into a DataFrame.
    
    Args:
        data_list: List of patient records with nested history
        
    Returns:
        pd.DataFrame: Flattened data with one row per visit
    """
    flat_data = []
    for entry in data_list:
        # Extract patient-level information (excluding history)
        patient_info = {key: entry[key] for key in entry if key != 'history'}
        
        # Flatten each visit in the patient's history
        for record in entry['history']:
            flat_record = {**patient_info, **record}
            flat_data.append(flat_record)
    
    return pd.DataFrame(flat_data)


def aggregate_patient_data(group):
    """
    Aggregate multiple visits for a single patient into one record.
    
    Args:
        group: DataFrame group for a single patient
        
    Returns:
        pd.Series: Aggregated patient data
    """
    # Combine all diagnoses into a unique set
    diagnoses_combined = set()
    for diagnosis_list in group['diagnoses']:
        if isinstance(diagnosis_list, list):
            diagnoses_combined.update(diagnosis_list)
    
    # Combine all abnormal labs into a unique set
    abnormal_labs_combined = set()
    for lab_list in group['abnormal_labs']:
        if isinstance(lab_list, list):
            abnormal_labs_combined.update(lab_list)
    
    # Determine CKD status (if any visit has CKD, mark as positive)
    next_ckd = 1 if group['next_ckd'].any() == 1 else 0
    num_visits = group.shape[0]
    
    # Create aggregated patient record
    aggregated_data = {
        'patient_id': group['patient_id'].iloc[0],
        'age': group['age'].iloc[0],
        'gender': group['gender'].iloc[0],
        'is_ckd': next_ckd,
        'diagnoses': list(diagnoses_combined),
        'abnormal_labs': list(abnormal_labs_combined),
        'num_visits': num_visits
    }
    
    return pd.Series(aggregated_data)


def create_patient_history(aggregated_df):
    """
    Create a text-based patient history string for each patient.
    
    Args:
        aggregated_df: DataFrame with aggregated patient data
        
    Returns:
        pd.DataFrame: DataFrame with added patient_history column
    """
    # Convert lists to strings
    aggregated_df['diagnoses_str'] = aggregated_df['diagnoses'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else str(x)
    )
    aggregated_df['abnormal_labs_str'] = aggregated_df['abnormal_labs'].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else str(x)
    )
    
    # Create comprehensive patient history string
    aggregated_df['patient_history'] = (
        "diagnoses:" + aggregated_df['diagnoses_str'] + " , " +
        aggregated_df['abnormal_labs_str'] + " " + "age:" +
        aggregated_df['age'].astype(str) + " , " + "gender:" +
        aggregated_df['gender'].astype(str) + " , " + "number of visits:" +
        aggregated_df['num_visits'].astype(str)
    )
    
    return aggregated_df


def process_dataset_split(data_list, split_name):
    """
    Process a single dataset split (train/val/test).
    
    Args:
        data_list: List of patient records
        split_name: Name of the split (for logging)
        
    Returns:
        pd.DataFrame: Processed and aggregated DataFrame
    """
    logger.info(f"Processing {split_name} split...")
    
    # Flatten the nested data structure
    logger.info(f"Flattening {split_name} data...")
    flat_df = flatten_patient_data(data_list)
    logger.info(f"Flattened {split_name}: {len(flat_df)} records")
    
    # Aggregate by patient
    logger.info(f"Aggregating {split_name} data by patient...")
    aggregated_df = flat_df.groupby('patient_id').apply(
        aggregate_patient_data, include_groups=False
    ).reset_index(drop=True)
    logger.info(f"Aggregated {split_name}: {len(aggregated_df)} patients")
    
    # Create patient history strings
    logger.info(f"Creating patient history strings for {split_name}...")
    aggregated_df = create_patient_history(aggregated_df)
    
    # Log class distribution
    ckd_positive = aggregated_df['is_ckd'].sum()
    ckd_negative = len(aggregated_df) - ckd_positive
    logger.info(f"{split_name} class distribution - Positive: {ckd_positive}, Negative: {ckd_negative}")
    
    return aggregated_df


def aggregate_data(input_dir, output_dir):
    """
    Main function to aggregate patient data from all splits.
    
    Args:
        input_dir: Directory containing the processed pickle files
        output_dir: Directory to save aggregated data
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load the processed data splits
    logger.info("Loading processed data splits...")
    
    try:
        train_data = pd.read_pickle(input_path / 'train.pkl')
        test_data = pd.read_pickle(input_path / 'test.pkl')
        val_data = pd.read_pickle(input_path / 'val.pkl')
        
        logger.info(f"Loaded train: {len(train_data)} samples")
        logger.info(f"Loaded test: {len(test_data)} samples")
        logger.info(f"Loaded val: {len(val_data)} samples")
        
    except FileNotFoundError as e:
        logger.error(f"Error loading data files: {e}")
        raise
    
    # Process each split
    splits = {
        'train': train_data,
        'test': test_data,
        'val': val_data
    }
    
    aggregated_splits = {}
    for split_name, data in splits.items():
        aggregated_splits[split_name] = process_dataset_split(data, split_name)
    
    # Save aggregated data
    logger.info("Saving aggregated data...")
    for split_name, aggregated_df in aggregated_splits.items():
        output_file = output_path / f"aggregated_{split_name}_df.pkl"
        aggregated_df.to_pickle(output_file)
        logger.info(f"Saved {split_name} to {output_file}")
    
    # Print summary statistics
    logger.info("\n=== AGGREGATION SUMMARY ===")
    for split_name, aggregated_df in aggregated_splits.items():
        total_patients = len(aggregated_df)
        ckd_positive = aggregated_df['is_ckd'].sum()
        ckd_negative = total_patients - ckd_positive
        avg_visits = aggregated_df['num_visits'].mean()
        
        logger.info(f"{split_name.upper()}:")
        logger.info(f"  Total patients: {total_patients}")
        logger.info(f"  CKD positive: {ckd_positive} ({ckd_positive/total_patients*100:.1f}%)")
        logger.info(f"  CKD negative: {ckd_negative} ({ckd_negative/total_patients*100:.1f}%)")
        logger.info(f"  Average visits per patient: {avg_visits:.1f}")
        logger.info("")
    
    return aggregated_splits


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Aggregate patient data for CKD prediction')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing processed pickle files (train.pkl, test.pkl, val.pkl)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save aggregated data files')
    
    args = parser.parse_args()
    
    # Run aggregation
    aggregated_data = aggregate_data(args.input_dir, args.output_dir)
    
    logger.info(f"Aggregation complete! Files saved to {args.output_dir}")


if __name__ == "__main__":
    main() 