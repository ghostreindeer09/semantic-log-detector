#!/usr/bin/env python3
"""
Main Demo Script - Demonstrates the Semantic Log Anomaly Detection System.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import AnomalyDetectionPipeline
from data.log_generator import LogGenerator


def main():
    print("=" * 60)
    print("🔍 Semantic Log Anomaly Detection System")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n📦 Initializing pipeline...")
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    pipeline = AnomalyDetectionPipeline(config_path)
    
    # Generate sample data
    print("\n📝 Generating sample log data...")
    generator = LogGenerator(seed=42)
    normal_logs, anomaly_logs = generator.generate_dataset(n_normal=200, n_anomaly=20)
    
    print(f"   Generated {len(normal_logs)} normal logs")
    print(f"   Generated {len(anomaly_logs)} anomaly logs")
    
    # Fit the pipeline on normal logs
    print("\n🎯 Training model on normal logs...")
    pipeline.fit(normal_logs)
    
    # Test detection on normal logs
    print("\n✅ Testing on normal logs...")
    normal_detected = 0
    for log in normal_logs[:20]:
        result = pipeline.detect(log)
        if result.anomaly_score.is_anomaly:
            normal_detected += 1
    print(f"   False positives: {normal_detected}/20")
    
    # Test detection on anomaly logs
    print("\n🚨 Testing on anomaly logs...")
    anomaly_detected = 0
    for i, log in enumerate(anomaly_logs[:10]):
        result = pipeline.detect(log)
        if result.anomaly_score.is_anomaly:
            anomaly_detected += 1
            if result.explanation:
                print(f"\n   Anomaly #{i+1}:")
                print(f"   Score: {result.anomaly_score.score:.2%}")
                print(f"   Summary: {result.explanation.summary}")
    
    print(f"\n   True positives: {anomaly_detected}/10")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Detection Summary")
    print("=" * 60)
    print(f"   Vector DB Size: {pipeline.vector_db.size()} logs")
    print(f"   False Positive Rate: {normal_detected/20:.1%}")
    print(f"   True Positive Rate: {anomaly_detected/10:.1%}")
    print("\n✨ Demo completed successfully!")
    
    return pipeline


if __name__ == "__main__":
    pipeline = main()
