"""
SIEM Dataset Loader - Loads the Advanced SIEM Dataset from Hugging Face.
"""

from datasets import load_dataset
from typing import List, Tuple, Optional, Dict
import pandas as pd


class SIEMDataLoader:
    """
    Loader for the Advanced SIEM Dataset from Hugging Face.
    
    Dataset: darkknight25/Advanced_SIEM_Dataset
    Contains security events with severity levels, descriptions, and raw logs.
    """
    
    SEVERITY_MAPPING = {
        'info': 0,
        'low': 1,
        'medium': 2,
        'high': 3,
        'critical': 4,
        'emergency': 5
    }
    
    # Consider high, critical, emergency as anomalies
    ANOMALY_SEVERITIES = {'high', 'critical', 'emergency'}
    
    def __init__(self, max_samples: Optional[int] = None):
        """
        Initialize the SIEM data loader.
        
        Args:
            max_samples: Maximum number of samples to load (None for all)
        """
        self.max_samples = max_samples
        self.dataset = None
        self.df = None
    
    def load(self) -> None:
        """Load the dataset from Hugging Face."""
        print("Loading Advanced SIEM Dataset from Hugging Face...")
        self.dataset = load_dataset(
            "darkknight25/Advanced_SIEM_Dataset",
            split="train"
        )
        
        if self.max_samples:
            self.dataset = self.dataset.select(range(min(self.max_samples, len(self.dataset))))
        
        print(f"Loaded {len(self.dataset)} records")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame."""
        if self.dataset is None:
            self.load()
        
        self.df = self.dataset.to_pandas()
        
        # Extract risk_score from advanced_metadata
        self.df['risk_score'] = self.df['advanced_metadata'].apply(
            lambda x: x.get('risk_score', 50) if isinstance(x, dict) else 50
        )
        
        # Create is_anomaly flag based on severity
        self.df['is_anomaly'] = self.df['severity'].isin(self.ANOMALY_SEVERITIES)
        
        return self.df
    
    def get_log_texts(self) -> List[str]:
        """Get log descriptions/texts for embedding."""
        if self.df is None:
            self.to_dataframe()
        
        # Use description as the main text for embedding
        return self.df['description'].tolist()
    
    def get_raw_logs(self) -> List[str]:
        """Get raw CEF-formatted logs."""
        if self.df is None:
            self.to_dataframe()
        
        return self.df['raw_log'].tolist()
    
    def split_normal_anomaly(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Split dataset into normal and anomaly logs.
        
        Returns:
            Tuple of (normal_descriptions, anomaly_descriptions, normal_ids, anomaly_ids)
        """
        if self.df is None:
            self.to_dataframe()
        
        normal_mask = ~self.df['is_anomaly']
        anomaly_mask = self.df['is_anomaly']
        
        normal_descriptions = self.df.loc[normal_mask, 'description'].tolist()
        anomaly_descriptions = self.df.loc[anomaly_mask, 'description'].tolist()
        normal_ids = self.df.loc[normal_mask, 'event_id'].tolist()
        anomaly_ids = self.df.loc[anomaly_mask, 'event_id'].tolist()
        
        print(f"Normal logs: {len(normal_descriptions)}")
        print(f"Anomaly logs: {len(anomaly_descriptions)}")
        
        return normal_descriptions, anomaly_descriptions, normal_ids, anomaly_ids
    
    def get_metadata(self) -> List[Dict]:
        """Get metadata for each log entry."""
        if self.df is None:
            self.to_dataframe()
        
        metadata = []
        for _, row in self.df.iterrows():
            metadata.append({
                'event_id': row['event_id'],
                'event_type': row['event_type'],
                'severity': row['severity'],
                'source': row['source'],
                'risk_score': row['risk_score'],
                'is_anomaly': row['is_anomaly'],
                'description': row['description'][:200]  # Truncate for storage
            })
        
        return metadata
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        if self.df is None:
            self.to_dataframe()
        
        return {
            'total_records': len(self.df),
            'event_types': self.df['event_type'].value_counts().to_dict(),
            'severity_distribution': self.df['severity'].value_counts().to_dict(),
            'anomaly_count': self.df['is_anomaly'].sum(),
            'normal_count': (~self.df['is_anomaly']).sum(),
            'avg_risk_score': self.df['risk_score'].mean()
        }


if __name__ == "__main__":
    # Test the loader
    loader = SIEMDataLoader(max_samples=1000)
    loader.load()
    
    stats = loader.get_stats()
    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    normal, anomaly, _, _ = loader.split_normal_anomaly()
    
    print(f"\n=== Sample Normal Log ===")
    print(normal[0] if normal else "No normal logs")
    
    print(f"\n=== Sample Anomaly Log ===")
    print(anomaly[0] if anomaly else "No anomaly logs")
