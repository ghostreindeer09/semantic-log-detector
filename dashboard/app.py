"""
Flask Dashboard for Anomaly Detection - Alerting & Visualization.
"""

import os
import sys
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
import yaml
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AnomalyDetectionPipeline

app = Flask(__name__, template_folder='templates', static_folder='static')
CORS(app)

pipeline = None
detection_history = []


@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Get pipeline status."""
    global pipeline
    
    db_size = 0
    if pipeline:
        if hasattr(pipeline, 'vector_db'):
            db_size = pipeline.vector_db.size()
        elif hasattr(pipeline, 'embeddings') and pipeline.embeddings is not None:
             db_size = len(pipeline.embeddings)
        elif hasattr(pipeline, 'ids') and pipeline.ids:
             db_size = len(pipeline.ids)
        elif hasattr(pipeline, 'texts') and pipeline.texts:
             db_size = len(pipeline.texts)

    return jsonify({
        'is_fitted': pipeline.is_fitted if pipeline else False,
        'vector_db_size': db_size,
        'history_count': len(detection_history)
    })


@app.route('/api/detect', methods=['POST'])
def detect():
    """Detect anomaly in a log."""
    global pipeline, detection_history
    
    if not pipeline or not pipeline.is_fitted:
        return jsonify({'error': 'Pipeline not fitted'}), 400
    
    data = request.json
    log = data.get('log', '')
    
    if not log:
        return jsonify({'error': 'No log provided'}), 400
    
    # Metadata for hybrid model (optional)
    metadata = {}
    
    result = pipeline.detect(log)
    
    # Handle dictionary result (from Hybrid pipeline)
    if isinstance(result, dict):
        response = {
            'log_id': result['log_id'],
            'cleaned_log': result['cleaned_log'],
            'score': result['score'],
            'is_anomaly': result['is_anomaly'],
            'confidence': result.get('confidence', 0.5), # Hybrid might not have explicit confidence yet
            'neighbors': result['neighbors']
        }
        if result.get('explanation'):
            response['explanation'] = {
                'summary': result['explanation'].summary,
                'details': result['explanation'].details,
                'recommendations': result['explanation'].recommendations
            }
    else:
        # Handle object result (from default pipeline)
        response = {
            'log_id': result.log_id,
            'cleaned_log': result.cleaned_log,
            'score': result.anomaly_score.score,
            'is_anomaly': result.anomaly_score.is_anomaly,
            'confidence': result.anomaly_score.confidence,
            'neighbors': result.anomaly_score.nearest_neighbors[:3]
        }
        
        if result.explanation:
            response['explanation'] = {
                'summary': result.explanation.summary,
                'details': result.explanation.details,
                'recommendations': result.explanation.recommendations
            }
    
    detection_history.append(response)
    return jsonify(response)


@app.route('/api/history')
def history():
    """Get detection history."""
    return jsonify({'history': detection_history[-50:]})


@app.route('/api/stats')
def stats():
    """Get detection statistics."""
    if not detection_history:
        return jsonify({'total': 0, 'anomalies': 0, 'rate': 0})
    
    anomalies = sum(1 for d in detection_history if d.get('is_anomaly'))
    return jsonify({
        'total': len(detection_history),
        'anomalies': anomalies,
        'rate': anomalies / len(detection_history)
    })


def init_app(config_path=None):
    """Initialize the application."""
    global pipeline
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'models', 'siem_model'
    )
    
    # Try to load the trained Hybrid pipeline first
    if os.path.exists(f"{model_path}.state"):
        print(f"Loading trained SIEM model from {model_path}...")
        try:
            # We need to import HybridAnomalyPipeline here to resolve circular imports if any,
            # or just pickle compatibility. Since it's in a different script, 
            # we might need to make sure the class is available.
            # However, looking at project structure, HybridAnomalyPipeline is in train_siem.py
            # which might not be importable if not in module path.
            # Ideally HybridAnomalyPipeline should be in src/pipeline.py or similar.
            
            # Let's import it dynamically
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from train_siem import HybridAnomalyPipeline
            pipeline = HybridAnomalyPipeline.load(model_path)
            print("✅ Trained SIEM model loaded successfully!")
            
            # Pre-load some history for visualization
            if not detection_history:
                print("Pre-loading sample history...")
                try:
                    # Provide some sample anomalies for the dashboard
                    samples = [
                        {
                            'log_id': 'evt_9283',
                            'cleaned_log': 'firewall deny tcp traffic from 192.168.1.100:445 to 10.0.0.5:135 map rule: 7',
                            'score': 0.88,
                            'is_anomaly': True,
                            'timestamp': (datetime.now() - timedelta(minutes=5)).isoformat(),
                            'explanation': {
                                'summary': 'High severity firewall anomaly detected.',
                                'recommendations': ['Check firewall rules', 'Investigate source IP']
                            }
                        },
                        {
                            'log_id': 'evt_9284', 
                            'cleaned_log': 'suspicious process powershell.exe started with encoded command line',
                            'score': 0.95,
                            'is_anomaly': True,
                            'timestamp': (datetime.now() - timedelta(minutes=12)).isoformat(),
                            'explanation': {
                                'summary': 'Critical endpoint security alert.',
                                'recommendations': ['Isolate host', 'Analyze process tree']
                            }
                        },
                         {
                            'log_id': 'evt_9285',
                            'cleaned_log': 'failed login attempt user=root src_ip=45.2.1.33 reason=bad_password',
                            'score': 0.72,
                            'is_anomaly': True,
                            'timestamp': (datetime.now() - timedelta(minutes=25)).isoformat(),
                            'explanation': {
                                'summary': 'Authentication anomaly detected.',
                                'recommendations': ['Check user account status', 'Verify IP reputation']
                            }
                        }
                    ]
                    detection_history.extend(samples)
                except Exception as e:
                    print(f"Error pre-loading history: {e}")
            
            return app
        except Exception as e:
            print(f"❌ Failed to load trained model: {e}")
            print("Falling back to default pipeline...")

    config_path = config_path or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'config', 'config.yaml'
    )
    pipeline = AnomalyDetectionPipeline(config_path)
    return app


if __name__ == '__main__':
    app = init_app()
    app.run(host='0.0.0.0', port=8080, debug=True)
