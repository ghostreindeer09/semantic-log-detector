#!/usr/bin/env python3
"""
Test Custom Data - Run the trained model on your own data.
"""

import sys
import os
import argparse
import pandas as pd
import json
from train_siem import HybridAnomalyPipeline

def load_data(file_path):
    """Load data from CSV or JSONL."""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.jsonl') or file_path.endswith('.json'):
        # Try both line-delimited and standard JSON
        try:
            return pd.read_json(file_path, lines=True)
        except ValueError:
            return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please use .csv or .json/.jsonl")

def main():
    parser = argparse.ArgumentParser(description="Run anomaly detection on custom data")
    parser.add_argument("file_path", help="Path to your data file (CSV or JSONL)")
    parser.add_argument("--text-col", default="description", help="Column containing the log text")
    parser.add_argument("--severity-col", default="severity", help="Column containing severity (optional)")
    parser.add_argument("--risk-col", default="risk_score", help="Column containing risk score (optional)")
    parser.add_argument("--label-col", help="Column containing ground truth (1=anomaly, 0=normal) for metrics")
    parser.add_argument("--model-path", default="models/siem_model", help="Path to saved model")
    
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"📦 Loading data from {args.file_path}...")
    try:
        df = load_data(args.file_path)
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
        
    print(f"   Loaded {len(df)} records")
    
    if args.text_col not in df.columns:
        print(f"❌ Error: Text column '{args.text_col}' not found in data.")
        print(f"   Available columns: {list(df.columns)}")
        return

    # 2. Load Model
    print(f"\n🧠 Loading trained model from {args.model_path}...")
    if not os.path.exists(f"{args.model_path}.state"):
        print(f"❌ Error: Model not found at {args.model_path}")
        print("   Please run 'python train_siem.py' first to train and save the model.")
        return
        
    try:
        pipeline = HybridAnomalyPipeline.load(args.model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    # 3. Prepare Metadata
    print("\n⚙️ Preparing data...")
    logs = df[args.text_col].fillna("").astype(str).tolist()
    metadata_list = []
    
    for _, row in df.iterrows():
        meta = {}
        if args.severity_col in row:
            meta['severity'] = row[args.severity_col]
        if args.risk_col in row:
            # Handle risk score normalization if needed, pipeline expects 0-100 usually
            meta['risk_score'] = row[args.risk_col]
        metadata_list.append(meta)

    # 4. Run Detection
    print("\n🔍 Running detection...")
    results = pipeline.detect_batch(logs, metadata_list)
    
    # Add results to dataframe
    df['anomaly_score'] = [r['score'] for r in results]
    df['is_anomaly_pred'] = [r['is_anomaly'] for r in results]
    
    # 5. Metrics (if label provided)
    if args.label_col and args.label_col in df.columns:
        print("\n📊 Calculating metrics...")
        y_true = df[args.label_col].astype(int)
        y_pred = df['is_anomaly_pred'].astype(int)
        
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        
        accuracy = (tp + tn) / len(df)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"   Accuracy:  {accuracy:.1%}")
        print(f"   Precision: {precision:.1%}")
        print(f"   Recall:    {recall:.1%}")
        print(f"   F1 Score:  {f1:.1%}")
        print(f"\n   TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    
    # 6. Show Anomalies
    anomalies = df[df['is_anomaly_pred'] == True]
    print(f"\n🚨 Detected {len(anomalies)} anomalies out of {len(df)} records")
    
    if len(anomalies) > 0:
        print("\n📋 Top 5 Anomalies:")
        top_anomalies = anomalies.sort_values('anomaly_score', ascending=False).head(5)
        for _, row in top_anomalies.iterrows():
            print(f"   Score: {row['anomaly_score']:.2%}")
            print(f"   Log: {str(row[args.text_col])[:100]}...")
            print("-" * 40)
            
    # Save results
    output_path = "detection_results.csv"
    df.to_csv(output_path, index=False)
    print(f"\n💾 Results saved to {output_path}")

if __name__ == "__main__":
    main()
