<!DOCTYPE html>
<html>
<head>
  <title>Semantic Log Detector</title>
</head>
<body>

<h1>Semantic Log Detector</h1>

<p>
A real‑time semantic log anomaly detection system that identifies suspicious or malicious log events using machine learning. Instead of simple text matching, it uses semantic embeddings to understand the meaning of log messages and detect unusual patterns.
</p>

<h2>Features</h2>
<ul>
  <li>Semantic log understanding using <strong>Sentence‑BERT</strong> embeddings</li>
  <li>Real‑time anomaly detection</li>
  <li>Efficient similarity search with <strong>Milvus</strong> or <strong>FAISS</strong></li>
  <li>Time‑aware scoring to detect sequential anomalies</li>
  <li>Human‑readable explanations of anomalies</li>
  <li>Promising performance on SIEM‑style datasets:
    <ul>
      <li>True Positive Rate: 100%</li>
      <li>True Negative Rate: 92%</li>
      <li>Overall Accuracy: 96%</li>
    </ul>
  </li>
</ul>

<h2>Prerequisites</h2>
<ul>
  <li>Python 3.10+</li>
  <li>PyTorch</li>
  <li>sentence‑transformers</li>
  <li>Milvus or FAISS</li>
  <li>scikit‑learn, pandas, numpy</li>
  <li>Optional: Kafka / Google Pub/Sub for streaming logs</li>
</ul>

<h2>Installation</h2>
<pre>
git clone https://github.com/ghostreindeer09/semantic-log-detector.git
cd semantic-log-detector
pip install -r requirements.txt
</pre>

<h2>Datasets</h2>
<ul>
  <li><a href="https://github.com/logpai/loghub">HDFS Logs</a></li>
  <li><a href="https://github.com/logpai/loghub">BGL Logs</a></li>
  <li><a href="https://huggingface.co/datasets/darkknight25/Advanced_SIEM_Dataset">SIEM‑style Dataset on Hugging Face</a></li>
</ul>

<h2>Screenshots</h2>
<p>Example screenshots of the system in action:</p>
<ul>
  <img width="1470" height="845" alt="Screenshot 2026-02-03 at 15 18 23" src="https://github.com/user-attachments/assets/173e415f-97a0-4177-85c7-ef714632da35" />
  <img width="1466" height="843" alt="Screenshot 2026-02-03 at 15 18 36" src="https://github.com/user-attachments/assets/3aa50c20-899e-4cbc-933d-ca5478693da1" />
  <img width="1468" height="846" alt="Screenshot 2026-02-03 at 15 18 51" src="https://github.com/user-attachments/assets/6a33b9e9-1273-4753-acf9-22fa0eb8df56" />


</ul>

<h2>Future Improvements</h2>
<ul>
  <li>Reduce false positives by refining thresholds and temporal modeling</li>
  <li>Add support for more log sources (cloud, IoT, containers)</li>
  <li>Integration with real‑time monitoring and alerting systems</li>
  <li>Hosted deployment using Vertex AI or similar platforms</li>
</ul>

<h2>License</h2>
<p>This project is licensed under the MIT License — see the <a href="LICENSE">LICENSE</a> file for details.</p>

<h2>Contact</h2>
<p>For contributions or questions, reach out to <strong>ghostreindeer09</strong> or open an issue on this repository.</p>

</body>
</html>
