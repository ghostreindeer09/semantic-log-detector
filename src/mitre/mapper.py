from typing import Dict, Optional

from ..core.schemas import MitreMapping


class MitreMapper:
    """
    Maps detection outputs to MITRE ATT&CK framework.

    Supports three mapping sources (in priority order):
    1. IDS classification — predicted attack class from LightGBM
    2. Rule ID — deterministic rule-engine match
    3. Keyword fallback — simple text matching on log content
    """

    def __init__(self):
        # ── Technique database (ID → (tactic, name)) ──
        self.technique_map = {
            # Original 5
            'T1110':     ('Credential Access', 'Brute Force'),
            'T1078':     ('Defense Evasion', 'Valid Accounts'),
            'T1059':     ('Execution', 'Command and Scripting Interpreter'),
            'T1046':     ('Discovery', 'Network Service Discovery'),
            'T1003':     ('Credential Access', 'OS Credential Dumping'),
            # Sub-techniques
            'T1110.001': ('Credential Access', 'Brute Force: Password Guessing'),
            'T1110.003': ('Credential Access', 'Brute Force: Password Spraying'),
            'T1059.001': ('Execution', 'PowerShell'),
            # Network-attack techniques (for CIC-IDS2017 classes)
            'T1498':     ('Impact', 'Network Denial of Service'),
            'T1498.001': ('Impact', 'Network DoS: Direct Network Flood'),
            'T1499':     ('Impact', 'Endpoint Denial of Service'),
            'T1499.003': ('Impact', 'Endpoint DoS: Application Exhaustion Flood'),
            'T1071':     ('Command and Control', 'Application Layer Protocol'),
            'T1190':     ('Initial Access', 'Exploit Public-Facing Application'),
            'T1595':     ('Reconnaissance', 'Active Scanning'),
            'T1595.001': ('Reconnaissance', 'Active Scanning: Scanning IP Blocks'),
        }

        # ── CIC-IDS2017 attack class → MITRE technique ──
        self.ids_class_mapping = {
            'DDoS':         'T1498.001',
            'DoS':          'T1499.003',
            'PortScan':     'T1595.001',
            'Brute Force':  'T1110.001',
            'Web Attack':   'T1190',
            'Bot':          'T1071',
            'Infiltration': 'T1078',
        }

        # ── Rule ID → technique (existing, kept) ──
        self.rule_mapping = {
            'R001': 'T1110',   # Failed Login Burst
            'R002': 'T1078',   # Privilege Escalation
            'R003': 'T1046',   # Suspicious IP scanning
        }

    # ---------------------------------------------------------------- #
    #  Primary interface (used by AlertBuilder)                         #
    # ---------------------------------------------------------------- #

    def map_detection(
        self,
        predicted_class: Optional[str] = None,
        rule_id: Optional[str] = None,
        log_text: str = "",
    ) -> MitreMapping:
        """
        Map a detection to MITRE ATT&CK using the best available signal.

        Priority: IDS classification > Rule ID > Keyword fallback.
        Returns a ``MitreMapping`` Pydantic model.
        """
        technique_id: Optional[str] = None
        confidence = 0.0

        # Priority 1: IDS classified attack type
        if predicted_class and predicted_class in self.ids_class_mapping:
            technique_id = self.ids_class_mapping[predicted_class]
            confidence = 0.9

        # Priority 2: Rule ID match
        elif rule_id and rule_id in self.rule_mapping:
            technique_id = self.rule_mapping[rule_id]
            confidence = 0.95

        # Priority 3: Keyword fallback
        else:
            technique_id, confidence = self._keyword_fallback(log_text)

        if technique_id and technique_id in self.technique_map:
            tactic, name = self.technique_map[technique_id]
            return MitreMapping(
                technique_id=technique_id,
                technique_name=name,
                tactic=tactic,
                confidence=confidence,
            )

        return MitreMapping()

    # ---------------------------------------------------------------- #
    #  Backward-compatible interface                                    #
    # ---------------------------------------------------------------- #

    def map_alert(self, rule_id: Optional[str] = None, log_text: str = "") -> Dict:
        """
        Legacy interface — returns a plain dict.

        Kept for backward compatibility with existing code that calls
        ``mitre_mapper.map_alert(rule_id=..., log_text=...)``.
        """
        mapping = self.map_detection(rule_id=rule_id, log_text=log_text)
        return {
            'mitre_technique_id': mapping.technique_id,
            'mitre_technique_name': mapping.technique_name,
            'mitre_tactic': mapping.tactic,
        }

    # ---------------------------------------------------------------- #
    #  Private                                                          #
    # ---------------------------------------------------------------- #

    def _keyword_fallback(self, log_text: str):
        """Simple keyword matching on log content."""
        if not log_text:
            return None, 0.0

        text = log_text.lower()
        if "brute" in text:
            return 'T1110', 0.5
        if "sudo" in text or "su root" in text or "privilege" in text:
            return 'T1078', 0.5
        if "cmd" in text or "powershell" in text:
            return 'T1059', 0.5
        if "scan" in text or "portscan" in text:
            return 'T1595', 0.4
        if "ddos" in text or "flood" in text:
            return 'T1498', 0.4
        if "denial" in text or "dos" in text:
            return 'T1499', 0.3

        return None, 0.0
