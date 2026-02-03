"""
Sample Log Generator - Generates realistic log data for testing.
"""

import random
from datetime import datetime, timedelta
from typing import List, Tuple


class LogGenerator:
    """Generates realistic log entries for testing."""
    
    SERVICES = ['auth-service', 'api-gateway', 'user-service', 'payment-service', 
                'notification-service', 'database-proxy', 'cache-manager']
    
    NORMAL_TEMPLATES = [
        "{ts} INFO [{service}] Request processed successfully in {ms}ms",
        "{ts} INFO [{service}] User {user_id} logged in from {ip}",
        "{ts} DEBUG [{service}] Cache hit for key: session_{user_id}",
        "{ts} INFO [{service}] Health check passed - all systems operational",
        "{ts} INFO [{service}] Database query executed in {ms}ms",
        "{ts} DEBUG [{service}] Processing request from {ip}",
        "{ts} INFO [{service}] Session created for user {user_id}",
        "{ts} INFO [{service}] API call completed: GET /api/users/{user_id}",
        "{ts} DEBUG [{service}] Connection pool status: {pool_size}/100 active",
        "{ts} INFO [{service}] Scheduled task completed successfully",
    ]
    
    ANOMALY_TEMPLATES = [
        "{ts} ERROR [{service}] Failed to authenticate user - invalid credentials after 5 attempts from {ip}",
        "{ts} CRITICAL [{service}] Database connection timeout after 30s - all retries exhausted",
        "{ts} ERROR [{service}] Memory usage at 98% - OOM imminent on node-{node_id}",
        "{ts} WARN [{service}] Unusual request pattern detected from {ip} - possible DDoS attack",
        "{ts} ERROR [{service}] SSL certificate validation failed for external API",
        "{ts} CRITICAL [{service}] Disk space critical: 99% used on /var/log",
        "{ts} ERROR [{service}] Unexpected null pointer exception in payment processing",
        "{ts} WARN [{service}] Rate limit exceeded: 10000 requests/min from {ip}",
        "{ts} ERROR [{service}] Cryptographic operation failed - entropy source exhausted",
        "{ts} CRITICAL [{service}] Core service unresponsive - initiating emergency failover",
        "{ts} ERROR [{service}] SQL injection attempt blocked from {ip}",
        "{ts} WARN [{service}] Unusual file system access pattern detected",
    ]
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.base_time = datetime.now()
    
    def _generate_ip(self) -> str:
        return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"
    
    def _generate_timestamp(self, offset_seconds: int = 0) -> str:
        ts = self.base_time + timedelta(seconds=offset_seconds)
        return ts.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'
    
    def generate_normal_log(self, offset: int = 0) -> str:
        template = random.choice(self.NORMAL_TEMPLATES)
        return template.format(
            ts=self._generate_timestamp(offset),
            service=random.choice(self.SERVICES),
            user_id=random.randint(1000, 9999),
            ip=self._generate_ip(),
            ms=random.randint(5, 200),
            pool_size=random.randint(10, 50)
        )
    
    def generate_anomaly_log(self, offset: int = 0) -> str:
        template = random.choice(self.ANOMALY_TEMPLATES)
        return template.format(
            ts=self._generate_timestamp(offset),
            service=random.choice(self.SERVICES),
            ip=self._generate_ip(),
            node_id=random.randint(1, 10)
        )
    
    def generate_dataset(
        self,
        n_normal: int = 500,
        n_anomaly: int = 50
    ) -> Tuple[List[str], List[str]]:
        """Generate a dataset of normal and anomaly logs."""
        normal_logs = [self.generate_normal_log(i) for i in range(n_normal)]
        anomaly_logs = [self.generate_anomaly_log(i) for i in range(n_anomaly)]
        return normal_logs, anomaly_logs


if __name__ == "__main__":
    gen = LogGenerator()
    normal, anomaly = gen.generate_dataset(10, 5)
    
    print("=== Normal Logs ===")
    for log in normal[:5]:
        print(log)
    
    print("\n=== Anomaly Logs ===")
    for log in anomaly:
        print(log)
