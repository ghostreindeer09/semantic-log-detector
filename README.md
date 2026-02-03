# Semantic Log Anomaly Detection

A machine learning system for detecting anomalies in log data using semantic embeddings (Sentence-BERT), Isolation Forest, and k-NN similarity search.

## 🎯 Performance

Trained on the [Advanced SIEM Dataset](https://huggingface.co/datasets/darkknight25/Advanced_SIEM_Dataset):

| Metric | Score |
|--------|-------|
| **True Negative Rate** | 92% |
| **True Positive Rate** | 100% |
| **Overall Accuracy** | 96% |

## 🏗️ Architecture

```
Log Source → Log Ingestion → Preprocessor → Encoder (Sentence-BERT)
                                               ↓
                                      Feature Extraction
                                     (Semantic + Metadata)
                                               ↓
                              ┌────────────────┴────────────────┐
                              ↓                                  ↓
                     k-NN Similarity                    Isolation Forest
                              ↓                                  ↓
                              └────────────────┬────────────────┘
                                               ↓
                                      Anomaly Scorer
                                               ↓
                                    Explanation Engine
                                               ↓
                                   Alerting & Dashboard
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd semantic-log-anomaly-detection
pip install -r requirements.txt
```

### 2. Train on SIEM Dataset

```bash
python train_siem.py
```

This will:
- Download the Advanced SIEM Dataset from Hugging Face
- Train the model on normal logs
- Evaluate on both normal and anomalous logs
- Show sample detections with explanations

### 3. Run the Demo (Optional)

```bash
python demo.py
```

### 4. Start the Dashboard (Optional)

```bash
python -m dashboard.app
# Open http://localhost:5000
```

## 📁 Project Structure

```
semantic-log-anomaly-detection/
├── config/
│   └── config.yaml          # Configuration file
├── src/
│   ├── preprocessor.py      # Log parsing and normalization
│   ├── encoder.py           # Sentence-BERT embeddings
│   ├── vector_db.py         # FAISS vector storage
│   ├── similarity_search.py # k-NN similarity search
│   ├── anomaly_scorer.py    # Distance-based scoring
│   ├── explanation_engine.py # Human-readable explanations
│   ├── time_aware.py        # Temporal context handling
│   └── pipeline.py          # Main orchestration
├── data/
│   ├── siem_dataset.py      # HuggingFace SIEM dataset loader
│   └── log_generator.py     # Sample data generator
├── dashboard/
│   ├── app.py               # Flask API
│   └── templates/
│       └── index.html       # Dashboard UI
├── train_siem.py            # Training script for SIEM dataset
├── demo.py                  # Demo with synthetic data
└── requirements.txt         # Dependencies
```

## 🔧 How It Works

### 1. Preprocessing
- Parses raw logs and extracts structured information
- Normalizes timestamps, IPs, and paths
- Cleans text for embedding generation

### 2. Feature Extraction
- **Semantic Features**: Sentence-BERT embeddings (384 dimensions)
- **Metadata Features**: Risk score, severity level

### 3. Anomaly Detection (Hybrid Approach)
- **k-NN Distance**: Measures semantic distance from normal logs
- **Isolation Forest**: Detects statistical outliers
- **Combined Score**: Weighted combination of both methods

### 4. Explanation Engine
- Generates human-readable explanations
- Provides severity levels and recommendations
- Shows similar normal logs for context

## 📊 Dataset

The model is trained on the [Advanced SIEM Dataset](https://huggingface.co/datasets/darkknight25/Advanced_SIEM_Dataset) which contains:

- 8 event types: firewall, ids_alert, auth, endpoint, network, cloud, iot, ai
- 6 severity levels: info, low, medium, high, critical, emergency
- CEF-formatted raw logs
- MITRE ATT&CK technique references
- Behavioral analytics for ~10% of records

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
model:
  encoder: "all-MiniLM-L6-v2"
  embedding_dim: 384
  device: "cpu"  # or "cuda" for GPU

anomaly:
  k_neighbors: 5
  threshold_percentile: 95
```

## 📝 API Usage

```python
from train_siem import HybridAnomalyPipeline
from data.siem_dataset import SIEMDataLoader

# Load and prepare data
loader = SIEMDataLoader(max_samples=3000)
loader.load()
normal_logs, anomaly_logs, normal_ids, anomaly_ids = loader.split_normal_anomaly()

# Initialize and train
pipeline = HybridAnomalyPipeline(k_neighbors=5)
pipeline.fit(normal_logs, normal_ids, loader.get_metadata())

# Detect anomalies
result = pipeline.detect(suspicious_log, metadata)
print(f"Score: {result['score']:.2%}")
print(f"Is Anomaly: {result['is_anomaly']}")
```

## 🔬 Model Details

| Component | Implementation |
|-----------|----------------|
| Encoder | Sentence-BERT (all-MiniLM-L6-v2) |
| Embedding Dim | 384 |
| k-NN | sklearn NearestNeighbors |
| Outlier Detection | Isolation Forest (100 trees) |
| Contamination | 10% |

## 📝 License

MIT License
