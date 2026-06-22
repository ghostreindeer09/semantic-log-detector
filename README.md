# From 99.97% F1 to Production Failure: An Investigation into Temporal Leakage, Distribution Shift, and Reliability in ML-Based Intrusion Detection

[![Status](https://img.shields.io/badge/status-research--grade-blue)](#)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-teal)](#)

This repository serves as a case study and experimental evaluation pipeline for machine learning (ML) models in intrusion detection systems (IDS). It documents the investigation of a flow-based LightGBM intrusion detector from initial laboratory training to its failure modes under temporal distribution shifts.

---

## 🎯 Motivation

Traditional ML-based intrusion detection research regularly reports near-perfect metrics (e.g., F1 > 99%) on public datasets. However, when these models are deployed in production, their performance degrades. 

This project demonstrates that **random stratified splits introduce temporal leakage**, allowing models to memorize specific attack session profiles rather than learning generalizable detection signatures. By reorganizing the evaluation chronologically and analyzing probability calibration, we show why standard decision boundaries fail in deployment and investigate why naive online adaptive controls can adapt to attackers rather than normal traffic.

---

## 🏗 Architecture

The system is designed as a hybrid microservice to ingest log events, extract network flow patterns, perform inference, and route alerts.

- **Serving Gateway**: Built with **FastAPI** to support asynchronous, low-latency log stream ingestion.
- **Hybrid Detection Pipeline**:
  - **LightGBM Classifier**: Operates on 78 numeric network flow features for high-throughput detection.
  - **Sentence-BERT**: Performs semantic embedding analysis on unstructured commands and log texts.
  - **Isolation Forest**: Unsupervised check for volumetric rate anomalies.
  - **Rule Engine**: Pattern-matching signatures for high-confidence detections.
- **MITRE ATT&CK Mapping**: Automatically enriches alerts with TTP tagging (e.g., T1110 for Brute Force, T1498 for DoS).
- **Drift Monitoring**: Uses statistical checks (KS tests and PSI) over feature distributions to flag concept drift.

For details, see the [System Architecture Document](docs/architecture.md).

---

## 🔬 The Experimental Journey

The model's development and debugging journey evolved through five distinct evaluation stages:

```
 Random Evaluation (Stage 1)
   │   F1 = 0.9974, ROC-AUC = 0.99998 (Illusory production readiness)
   ▼
 Chronological Validation (Stage 2)
   │   F1 = 0.0563, Recall = 2.90% at default threshold 0.5
   ▼
 Threshold Tuning (Stage 3)
   │   Threshold 0.0012 recovers Recall to 99.56% but at 17.64% FPR
   ▼
 Probability Calibration (Stage 4)
   │   Platt & Isotonic calibrators fail due to validation-test shift
   ▼
 Adaptive Thresholding (Stage 5)
       Alert volume reduced by 92.6%, but recall collapses to 8.33%
```

1. **Random Evaluation**: A stratified 70/15/15 split of the cleaned CIC-IDS2017 dataset produced an F1 score of `0.9974` and an ROC-AUC of `0.99998`.
2. **Chronological Validation**: Evaluating the model temporally (training on Monday–Wednesday, validating on Thursday, and testing on Friday) caused recall to collapse to `2.90%` at the default threshold (0.5), as the model encountered unseen attack categories.
3. **Threshold Study**: Swapping to Youden's J optimal threshold of `0.0012` recovered recall to `99.56%` at the cost of a `17.64%` False Positive Rate.
4. **Probability Calibration**: Platt Scaling and Isotonic Regression failed to generalize from Thursday's validation split to Friday's test split due to distribution shift.
5. **Drift-Aware Adaptive Thresholding**: Quantile-based dynamic thresholds reduced alert volumes by `92.6%` compared to the fixed `0.001` baseline, but recall collapsed to `8.33%` as the model normalized attack traffic as the new baseline.

For the full phase-by-phase breakdown and empirical data, see the [Evaluation Journey Document](docs/evaluation_journey.md).

---

## 💡 Key Findings

- **Random splits overestimate performance**: Same-burst and same-session traffic leakage inflates offline performance.
- **High ROC-AUC does not guarantee a usable threshold**: The model ranks attack flows above benign ones (ROC-AUC = 0.93), but prediction probabilities are compressed near zero, rendering the default 0.5 threshold ineffective.
- **Calibration fails under distribution shift**: Platt scaling and Isotonic regression do not transfer to new distributions when the validation data itself is shifted.
- **Adaptive controls can normalize malicious behavior**: Quantile-based threshold adjustments can mistake persistent attack campaigns for normal baseline traffic, adapting *away* from threats.

For deep-dives and design suggestions, see the [Deployment Lessons & Recommendations Document](docs/deployment_lessons.md).

---

## ⚙️ Reproducibility

Execute the following commands to reproduce each stage of the validation pipeline:

### 1. Chronological Split Validation
Train the LightGBM classifier on Mon–Wed, validate on Thu, and evaluate on Fri:
```bash
python scripts/run_chronological_validation.py
```
*Outputs generated under `outputs/chronological_eval/` and `outputs/reports/chronological_threshold_report.md`.*

### 2. Probability Calibration Study
Sweep decision thresholds and train Platt and Isotonic calibrators on validation data:
```bash
python scripts/run_calibration_study.py
```
*Outputs generated under `outputs/calibration/` and `outputs/reports/calibration_study.md`.*

### 3. Adaptive Thresholding Study
Simulate streaming evaluation on Friday's test set using quantile and drift-triggered threshold controllers:
```bash
python scripts/run_adaptive_threshold_study.py
```
*Outputs generated under `outputs/adaptive_threshold/` and `outputs/reports/adaptive_threshold_design.md`.*

---

## 👨‍💻 Authors

**Rishit Sharma, Kokkula Srinivas**  
Detection Engineering | ML for Cyber Defense
