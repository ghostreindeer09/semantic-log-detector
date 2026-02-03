"""
Log Preprocessor Module

Handles parsing and normalization of raw log data.
"""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import hashlib


@dataclass
class ParsedLog:
    """Represents a parsed and normalized log entry."""
    raw_text: str
    cleaned_text: str
    timestamp: Optional[datetime] = None
    log_level: Optional[str] = None
    source: Optional[str] = None
    message: str = ""
    metadata: Dict = field(default_factory=dict)
    log_id: str = ""
    
    def __post_init__(self):
        if not self.log_id:
            self.log_id = hashlib.md5(
                f"{self.raw_text}{self.timestamp}".encode()
            ).hexdigest()[:12]


class LogPreprocessor:
    """
    Preprocessor for parsing and normalizing log entries.
    
    Handles various log formats and extracts structured information
    while cleaning the text for embedding generation.
    """
    
    # Common log patterns
    TIMESTAMP_PATTERNS = [
        r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?',
        r'\d{2}/\w{3}/\d{4}:\d{2}:\d{2}:\d{2}',
        r'\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}',
        r'\d{10,13}',  # Unix timestamp
    ]
    
    IP_PATTERN = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    PATH_PATTERN = r'(?:/[\w\-\.]+)+/?'
    HEX_PATTERN = r'\b0x[0-9a-fA-F]+\b'
    UUID_PATTERN = r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b'
    
    LOG_LEVELS = ['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'CRITICAL', 'FATAL']
    
    def __init__(
        self,
        max_length: int = 512,
        remove_timestamps: bool = True,
        remove_ips: bool = True,
        remove_paths: bool = False,
        lowercase: bool = True
    ):
        """
        Initialize the preprocessor.
        
        Args:
            max_length: Maximum length of cleaned text
            remove_timestamps: Whether to remove timestamps
            remove_ips: Whether to remove IP addresses
            remove_paths: Whether to remove file paths
            lowercase: Whether to convert to lowercase
        """
        self.max_length = max_length
        self.remove_timestamps = remove_timestamps
        self.remove_ips = remove_ips
        self.remove_paths = remove_paths
        self.lowercase = lowercase
        
        # Compile regex patterns
        self.timestamp_regex = [re.compile(p) for p in self.TIMESTAMP_PATTERNS]
        self.ip_regex = re.compile(self.IP_PATTERN)
        self.path_regex = re.compile(self.PATH_PATTERN)
        self.hex_regex = re.compile(self.HEX_PATTERN)
        self.uuid_regex = re.compile(self.UUID_PATTERN)
        self.log_level_regex = re.compile(
            r'\b(' + '|'.join(self.LOG_LEVELS) + r')\b',
            re.IGNORECASE
        )
        
    def extract_timestamp(self, text: str) -> Tuple[Optional[datetime], str]:
        """
        Extract timestamp from log text.
        
        Returns:
            Tuple of (datetime object or None, text with timestamp removed)
        """
        for pattern in self.timestamp_regex:
            match = pattern.search(text)
            if match:
                timestamp_str = match.group()
                try:
                    # Try parsing common formats
                    for fmt in [
                        '%Y-%m-%dT%H:%M:%S',
                        '%Y-%m-%d %H:%M:%S',
                        '%d/%b/%Y:%H:%M:%S',
                    ]:
                        try:
                            timestamp = datetime.strptime(timestamp_str[:19], fmt)
                            remaining_text = text[:match.start()] + text[match.end():]
                            return timestamp, remaining_text.strip()
                        except ValueError:
                            continue
                    
                    # Try Unix timestamp
                    if timestamp_str.isdigit():
                        ts = int(timestamp_str)
                        if ts > 1e12:  # Milliseconds
                            ts = ts / 1000
                        timestamp = datetime.fromtimestamp(ts)
                        remaining_text = text[:match.start()] + text[match.end():]
                        return timestamp, remaining_text.strip()
                        
                except (ValueError, OSError):
                    pass
                    
        return None, text
    
    def extract_log_level(self, text: str) -> Tuple[Optional[str], str]:
        """
        Extract log level from text.
        
        Returns:
            Tuple of (log level or None, text with log level removed)
        """
        match = self.log_level_regex.search(text)
        if match:
            level = match.group().upper()
            if level == 'WARN':
                level = 'WARNING'
            remaining_text = text[:match.start()] + text[match.end():]
            return level, remaining_text.strip()
        return None, text
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize log text for embedding.
        
        Args:
            text: Raw log text
            
        Returns:
            Cleaned text
        """
        cleaned = text
        
        # Remove timestamps if configured
        if self.remove_timestamps:
            for pattern in self.timestamp_regex:
                cleaned = pattern.sub('<TIMESTAMP>', cleaned)
        
        # Remove IPs if configured
        if self.remove_ips:
            cleaned = self.ip_regex.sub('<IP>', cleaned)
        
        # Remove paths if configured
        if self.remove_paths:
            cleaned = self.path_regex.sub('<PATH>', cleaned)
        
        # Normalize hex values
        cleaned = self.hex_regex.sub('<HEX>', cleaned)
        
        # Normalize UUIDs
        cleaned = self.uuid_regex.sub('<UUID>', cleaned)
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove special characters but keep meaningful punctuation
        cleaned = re.sub(r'[^\w\s\-\.\:\<\>\[\]\(\)]', '', cleaned)
        
        # Lowercase if configured
        if self.lowercase:
            cleaned = cleaned.lower()
        
        # Truncate to max length
        cleaned = cleaned[:self.max_length].strip()
        
        return cleaned
    
    def parse_log(self, raw_log: str, source: Optional[str] = None) -> ParsedLog:
        """
        Parse a single log entry.
        
        Args:
            raw_log: Raw log text
            source: Optional source identifier
            
        Returns:
            ParsedLog object
        """
        # Extract timestamp
        timestamp, text = self.extract_timestamp(raw_log)
        
        # Extract log level
        log_level, text = self.extract_log_level(text)
        
        # Clean the text for embedding
        cleaned_text = self.clean_text(text)
        
        return ParsedLog(
            raw_text=raw_log,
            cleaned_text=cleaned_text,
            timestamp=timestamp,
            log_level=log_level,
            source=source,
            message=text.strip(),
            metadata={
                'original_length': len(raw_log),
                'cleaned_length': len(cleaned_text),
            }
        )
    
    def parse_logs(
        self,
        raw_logs: List[str],
        source: Optional[str] = None
    ) -> List[ParsedLog]:
        """
        Parse multiple log entries.
        
        Args:
            raw_logs: List of raw log texts
            source: Optional source identifier
            
        Returns:
            List of ParsedLog objects
        """
        return [self.parse_log(log, source) for log in raw_logs]
    
    def parse_log_file(
        self,
        filepath: str,
        source: Optional[str] = None
    ) -> List[ParsedLog]:
        """
        Parse logs from a file.
        
        Args:
            filepath: Path to log file
            source: Optional source identifier (defaults to filename)
            
        Returns:
            List of ParsedLog objects
        """
        if source is None:
            source = filepath.split('/')[-1]
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            raw_logs = [line.strip() for line in f if line.strip()]
        
        return self.parse_logs(raw_logs, source)


if __name__ == "__main__":
    # Test the preprocessor
    test_logs = [
        "2024-01-15T10:23:45.123Z ERROR [UserService] Failed to authenticate user 192.168.1.100 - Invalid credentials",
        "Jan 15 10:23:46 app-server kernel: [12345.678] TCP connection from 10.0.0.1 dropped",
        "INFO 2024-01-15 10:23:47 Database query executed in 0.234s for /api/users/123",
        "[WARNING] Memory usage at 85% on node-1, consider scaling",
    ]
    
    preprocessor = LogPreprocessor()
    
    for log in test_logs:
        parsed = preprocessor.parse_log(log)
        print(f"\nRaw: {parsed.raw_text}")
        print(f"Cleaned: {parsed.cleaned_text}")
        print(f"Level: {parsed.log_level}")
        print(f"Timestamp: {parsed.timestamp}")
