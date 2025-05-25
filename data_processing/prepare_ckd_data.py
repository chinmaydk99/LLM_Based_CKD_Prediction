#!/usr/bin/env python3
"""
MIMIC-IV CKD Data Preparation Script

This script processes MIMIC-IV data to create sequential patient histories
for Chronic Kidney Disease (CKD) prediction. It converts ICD codes to CCSR,
processes lab results, and creates train/validation/test splits.

Usage:
    python prepare_ckd_data.py --mimic_dir /path/to/mimic --output_dir /path/to/output
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from icdmappings import Mapper
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
import warnings
import argparse

warnings.filterwarnings('ignore')


class MIMICSequentialProcessor:
    def __init__(self, mimic_dir):
        """Initialize processor with MIMIC-IV data directory"""
        self.mimic_dir = Path(mimic_dir)
        self.mapper = Mapper()

        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)

        # Define relevant lab tests
        self.ckd_relevant_labs = {
            'Creatinine', 'eGFR', 'BUN', 'Sodium', 'Potassium',
            'Chloride', 'Bicarbonate', 'Calcium', 'Phosphate',
            'Albumin', 'Hemoglobin', 'Platelets'
        }

        self.ckd_ccsr_code = 'GEN003'

    def load_raw_data(self):
        """Load and perform initial processing of MIMIC-IV tables"""
        self.logger.info("Loading MIMIC-IV data files...")

        try:
            # Load core tables
            self.admissions = pd.read_csv(self.mimic_dir / 'admissions.csv')
            self.patients = pd.read_csv(self.mimic_dir / 'patients.csv')
            self.diagnoses = pd.read_csv(self.mimic_dir / 'diagnoses_icd.csv')
            self.d_icd = pd.read_csv(self.mimic_dir / 'd_icd_diagnoses.csv')
            self.d_labitems = pd.read_csv(self.mimic_dir / 'd_labitems.csv')
            self.labevents = pd.read_csv(self.mimic_dir / 'labevents.csv')

            # Convert timestamps
            self.admissions['admittime'] = pd.to_datetime(self.admissions['admittime'])
            self.admissions['dischtime'] = pd.to_datetime(self.admissions['dischtime'])
            self.labevents['charttime'] = pd.to_datetime(self.labevents['charttime'])

            # Calculate length of stay
            self.admissions['length_of_stay'] = (
                self.admissions['dischtime'] - self.admissions['admittime']
            ).dt.total_seconds() / (24 * 60 * 60)

            self.logger.info("Data loading completed successfully")

        except FileNotFoundError as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    def convert_icd_to_ccsr(self, code, version):
        """
        Convert ICD code to CCSR
        For ICD-9: First convert to ICD-10, then to CCSR
        For ICD-10: Directly convert to CCSR
        """
        try:
            if version == 9:
                # Two-step conversion for ICD-9
                icd10_code = self.mapper.map(code, source='icd9', target='icd10')
                return self.mapper.map(icd10_code, source='icd10', target='ccsr')
            elif version == 10:
                # Direct conversion for ICD-10
                return self.mapper.map(code, source='icd10', target='ccsr')
            else:
                return "Invalid"
        except Exception as e:
            self.logger.warning(f"Error mapping code {code} (version {version}): {str(e)}")
            return "Invalid"

    def map_diagnoses_to_ccsr(self):
        """Map ICD codes to CCSR codes and merge with descriptions"""
        self.logger.info("Mapping diagnoses to CCSR codes...")

        diagnoses_processed = self.diagnoses.copy()

        # Add CCSR codes with proper version handling
        def map_single_diagnosis(row):
            if pd.isna(row['icd_version']):
                return "Invalid"
            return self.convert_icd_to_ccsr(row['icd_code'], row['icd_version'])

        tqdm.pandas(desc="Mapping ICD to CCSR")
        diagnoses_processed['ccsr_code'] = diagnoses_processed.progress_apply(
            map_single_diagnosis,
            axis=1
        )

        # Add diagnostic descriptions
        diagnoses_processed = diagnoses_processed.merge(
            self.d_icd[['icd_code', 'long_title']],
            on='icd_code',
            how='left'
        )

        # Log mapping statistics
        total_codes = len(diagnoses_processed)
        valid_mappings = len(diagnoses_processed[diagnoses_processed['ccsr_code'] != "Invalid"])

        self.logger.info(f"ICD to CCSR mapping completed:")
        self.logger.info(f"Total codes: {total_codes}")
        self.logger.info(f"Valid mappings: {valid_mappings}")
        self.logger.info(f"Invalid/failed mappings: {total_codes - valid_mappings}")

        return diagnoses_processed

    def process_lab_results(self):
        """Process lab results to identify abnormal values"""
        self.logger.info("Processing lab results...")

        relevant_labs = self.labevents.merge(
            self.d_labitems[['itemid', 'label']],
            on='itemid'
        )

        relevant_labs = relevant_labs[
            relevant_labs['label'].isin(self.ckd_relevant_labs)
        ]

        relevant_labs['abnormal'] = relevant_labs['flag'].fillna('').str.upper().isin(
            ['ABNORMAL', 'HIGH', 'LOW', 'CRITICAL']
        ).astype(int)

        self.logger.info(f"Processed {len(relevant_labs)} relevant lab results")
        return relevant_labs

    def is_valid_patient(self, subject_id, processed_diagnoses):
        """
        Check if patient meets inclusion criteria:
        - Has more than one visit
        - No CKD diagnosis in first visit
        """
        patient_visits = self.admissions[
            self.admissions['subject_id'] == subject_id
        ].sort_values('admittime')

        # Check visit count
        if len(patient_visits) < 2:
            return False

        # Check first visit for CKD
        first_visit_diagnoses = processed_diagnoses[
            processed_diagnoses['hadm_id'] == patient_visits.iloc[0]['hadm_id']
        ]

        has_ckd_first_visit = (
            first_visit_diagnoses['ccsr_code'] == self.ckd_ccsr_code
        ).any()

        return not has_ckd_first_visit

    def create_patient_history_samples(self, subject_id, processed_diagnoses, processed_labs):
        """Create sequential history samples for a patient"""
        # Get patient's visits in chronological order
        patient_visits = self.admissions[
            self.admissions['subject_id'] == subject_id
        ].sort_values('admittime')

        history_samples = []
        cumulative_history = []

        for idx, visit in patient_visits.iterrows():
            # Get diagnoses for this visit
            visit_diagnoses = processed_diagnoses[
                processed_diagnoses['hadm_id'] == visit['hadm_id']
            ]['long_title'].tolist()

            # Get lab results for this visit
            visit_labs = processed_labs[
                (processed_labs['hadm_id'] == visit['hadm_id']) &
                (processed_labs['abnormal'] == 1)
            ]['label'].unique().tolist()

            # Create visit record
            current_visit = {
                'diagnoses': visit_diagnoses,
                'abnormal_labs': visit_labs,
                'length_of_stay': visit['length_of_stay'],
                'days_since_first': (
                    visit['admittime'] - patient_visits['admittime'].min()
                ).days
            }

            # Add to cumulative history
            cumulative_history.append(current_visit)

            # Check for next visit
            future_visits = patient_visits[
                patient_visits['admittime'] > visit['admittime']
            ]

            if len(future_visits) == 0:  # No more visits
                continue

            next_visit = future_visits.iloc[0]

            # Check if CKD appears in next visit
            next_visit_diagnoses = processed_diagnoses[
                processed_diagnoses['hadm_id'] == next_visit['hadm_id']
            ]

            next_ckd = int(
                (next_visit_diagnoses['ccsr_code'] == self.ckd_ccsr_code).any()
            )

            # Create sample
            patient = self.patients[
                self.patients['subject_id'] == subject_id
            ].iloc[0]

            sample = {
                'patient_id': subject_id,
                'history': cumulative_history.copy(),
                'age': patient['anchor_age'],
                'gender': patient['gender'],
                'next_ckd': next_ckd
            }

            history_samples.append(sample)

            # Stop if CKD is diagnosed
            if next_ckd == 1:
                break

        return history_samples

    def process_and_save_data(self, output_dir):
        """Process MIMIC data and save train/val/test splits"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and process data
        self.load_raw_data()
        processed_diagnoses = self.map_diagnoses_to_ccsr()
        processed_labs = self.process_lab_results()

        # Filter valid patients
        all_patients = self.patients['subject_id'].unique()
        valid_patients = [
            pid for pid in tqdm(all_patients, desc="Filtering patients")
            if self.is_valid_patient(pid, processed_diagnoses)
        ]

        self.logger.info(f"Found {len(valid_patients)} valid patients out of {len(all_patients)} total")

        # Create dataset from valid patients
        dataset = []
        for subject_id in tqdm(valid_patients, desc="Processing valid patients"):
            patient_samples = self.create_patient_history_samples(
                subject_id, processed_diagnoses, processed_labs
            )
            dataset.extend(patient_samples)

        # Split by patient ID
        patient_ids = list(set(item['patient_id'] for item in dataset))
        train_patients, temp_patients = train_test_split(
            patient_ids, test_size=0.3, random_state=42
        )
        val_patients, test_patients = train_test_split(
            temp_patients, test_size=0.5, random_state=42
        )

        # Create splits
        train_data = [item for item in dataset if item['patient_id'] in train_patients]
        val_data = [item for item in dataset if item['patient_id'] in val_patients]
        test_data = [item for item in dataset if item['patient_id'] in test_patients]

        # Save splits
        for split_name, split_data in [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data)
        ]:
            with open(output_dir / f'{split_name}.pkl', 'wb') as f:
                pickle.dump(split_data, f)

        # Print statistics
        self.logger.info("\nDataset Statistics:")
        self.logger.info(f"Total samples: {len(dataset)}")
        self.logger.info(f"Train samples: {len(train_data)} from {len(train_patients)} patients")
        self.logger.info(f"Val samples: {len(val_data)} from {len(val_patients)} patients")
        self.logger.info(f"Test samples: {len(test_data)} from {len(test_patients)} patients")

        pos_train = sum(1 for x in train_data if x['next_ckd'] == 1)
        pos_val = sum(1 for x in val_data if x['next_ckd'] == 1)
        pos_test = sum(1 for x in test_data if x['next_ckd'] == 1)

        self.logger.info("\nClass Distribution:")
        self.logger.info(f"Train - Positive: {pos_train}, Negative: {len(train_data)-pos_train}")
        self.logger.info(f"Val - Positive: {pos_val}, Negative: {len(val_data)-pos_val}")
        self.logger.info(f"Test - Positive: {pos_test}, Negative: {len(test_data)-pos_test}")

        return train_data, val_data, test_data


def process_data(mimic_dir, output_dir):
    """Main execution function"""
    # Create processor and run pipeline
    processor = MIMICSequentialProcessor(mimic_dir)
    train_data, val_data, test_data = processor.process_and_save_data(output_dir)

    # Example: Print first sample from training data
    if train_data:
        print("\nExample training sample:")
        sample = train_data[0]
        print(f"Patient ID: {sample['patient_id']}")
        print(f"Age: {sample['age']}")
        print(f"Gender: {sample['gender']}")
        print(f"Next CKD: {sample['next_ckd']}")
        print(f"Number of visits in history: {len(sample['history'])}")
        print("\nFirst visit details:")
        print(sample['history'][0])

    return train_data, val_data, test_data


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='Process MIMIC-IV data for CKD prediction')
    parser.add_argument('--mimic_dir', type=str, required=True,
                        help='Path to MIMIC-IV data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to output directory for processed data')
    
    args = parser.parse_args()
    
    # Process the data
    train_data, val_data, test_data = process_data(args.mimic_dir, args.output_dir)
    
    print(f"\nProcessing complete! Data saved to {args.output_dir}")


if __name__ == "__main__":
    main() 