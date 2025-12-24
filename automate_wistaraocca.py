"""
automate_wistaraocca.py
========================

Script automation preprocessing steps di notebook experiment:
1. Load raw data
2. Hapus duplicates
3. Scale features (Amount dan Time)
4. Split data ke training dan test sets
5. Apply SMOTE
6. Save preprocessed data

Author: I Gede Abhijana Prayata Wistara
Dicoding Username: wistaraocca
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import warnings

warnings.filterwarnings('ignore')


def load_data(file_path: str) -> pd.DataFrame:

    print("=" * 67) # LOL
    print("STEP Ke-1: Loading Data")
    print("=" * 67)
    
    df = pd.read_csv(file_path)
    
    print(f"   Data loaded successfully!")
    print(f"   Shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 67)
    print("STEP Ke-2: Removing Duplicates")
    print("=" * 67)
    
    initial_shape = df.shape[0]
    df_clean = df.drop_duplicates()
    final_shape = df_clean.shape[0]
    
    print(f"   Duplicates removed!")
    print(f"   Before: {initial_shape:,} rows")
    print(f"   After: {final_shape:,} rows")
    print(f"   Removed: {initial_shape - final_shape:,} rows")
    
    return df_clean


def scale_features(df: pd.DataFrame) -> pd.DataFrame:

    print("\n" + "=" * 67)
    print("STEP Ke-3: Scaling Features")
    print("=" * 67)
    
    df_scaled = df.copy()
    scaler = StandardScaler()
    
    # Scale Amount
    df_scaled['Amount_scaled'] = scaler.fit_transform(df_scaled[['Amount']])
    
    # Scale Time
    df_scaled['Time_scaled'] = scaler.fit_transform(df_scaled[['Time']])
    
    # Drop original columns
    df_scaled = df_scaled.drop(['Amount', 'Time'], axis=1)
    
    print(f"   Features scaled!")
    print(f"   - Amount → Amount_scaled (StandardScaler)")
    print(f"   - Time → Time_scaled (StandardScaler)")
    print(f"   New shape: {df_scaled.shape}")
    
    return df_scaled


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):

    print("\n" + "=" * 67)
    print("STEP Ke-4: Splitting Data")
    print("=" * 67)
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"   Data split completed!")
    print(f"   Training set: {X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%)")
    print(f"   Test set: {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
    print(f"\n   Training class distribution:")
    print(f"   - Class 0: {(y_train == 0).sum():,}")
    print(f"   - Class 1: {(y_train == 1).sum():,}")
    
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state: int = 42):

    print("\n" + "=" * 67)
    print("STEP Ke-5: Applying SMOTE")
    print("=" * 67)
    
    print(f"Before SMOTE:")
    print(f"   - Class 0: {(y_train == 0).sum():,}")
    print(f"   - Class 1: {(y_train == 1).sum():,}")
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"\n  SMOTE applied!")
    print(f"After SMOTE:")
    print(f"   - Class 0: {(y_resampled == 0).sum():,}")
    print(f"   - Class 1: {(y_resampled == 1).sum():,}")
    print(f"   Total samples: {len(y_resampled):,}")
    
    return X_resampled, y_resampled


def save_preprocessed_data(X_train, y_train, X_test, y_test, 
                           output_dir: str = "preprocessing",
                           feature_columns: list = None):

    print("\n" + "=" * 67)
    print("STEP 6: Saving Preprocessed Data")
    print("=" * 67)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrames
    train_df = pd.DataFrame(X_train, columns=feature_columns)
    train_df['Class'] = y_train
    
    test_df = pd.DataFrame(X_test, columns=feature_columns)
    test_df['Class'] = y_test if isinstance(y_test, np.ndarray) else y_test.values
    
    # Save to CSV
    train_path = os.path.join(output_dir, "creditcard_train.csv")
    test_path = os.path.join(output_dir, "creditcard_test.csv")
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"   Data saved successfully!")
    print(f"   Training data: {train_path} ({train_df.shape[0]:,} rows)")
    print(f"   Test data: {test_path} ({test_df.shape[0]:,} rows)")
    
    return train_path, test_path


def preprocess_pipeline(input_path: str, output_dir: str = "preprocessing",
                        test_size: float = 0.2, random_state: int = 42):

    print("\n" + "=" * 67)
    print("CREDIT CARD FRAUD DETECTION - PREPROCESSING PIPELINE")
    print("=" * 67)
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 67)
    
    # Step Ein: Load data
    df = load_data(input_path)
    
    # Step Zwei: Remove duplicates
    df_clean = remove_duplicates(df)
    
    # Step Drei: Scale features
    df_scaled = scale_features(df_clean)
    
    # Step Polizei: Split data
    X_train, X_test, y_train, y_test = split_data(
        df_scaled, test_size=test_size, random_state=random_state
    )
    
    # Step Gestapo: Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(
        X_train, y_train, random_state=random_state
    )
    
    # Step Stasi: Save preprocessed data
    feature_columns = X_train.columns.tolist()
    train_path, test_path = save_preprocessed_data(
        X_train_resampled, y_train_resampled,
        X_test, y_test,
        output_dir=output_dir,
        feature_columns=feature_columns
    )

    print("\n" + "=" * 67)
    print("PREPROCESSING COMPLETED")
    print("=" * 67)
    print(f"""
   - Original data: {df.shape[0]:,} rows
   - After removing duplicates: {df_clean.shape[0]:,} rows
   - Training samples (after SMOTE): {len(y_train_resampled):,}
   - Test samples: {len(y_test):,}

   Output files:
   - {train_path}
   - {test_path}

   Sudah siap untuk model training.
    """)
    
    return {
        'X_train': X_train_resampled,
        'y_train': y_train_resampled,
        'X_test': X_test,
        'y_test': y_test,
        'train_path': train_path,
        'test_path': test_path,
        'feature_columns': feature_columns
    }


if __name__ == "__main__":
    result = preprocess_pipeline(
        input_path="creditcarddata_raw/creditcard.csv",
        output_dir="preprocessing",
        test_size=0.2,
        random_state=42
    )
