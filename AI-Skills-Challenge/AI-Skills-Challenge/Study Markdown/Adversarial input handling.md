
# Adversarial input handling

## 3. Adversarial Input Detection {#adversarial-input-detection}

Beyond prompt injection, adversarial inputs include any crafted input designed to cause unexpected, harmful, or policy-violating behavior — including jailbreaks, model inversion attacks, and data extraction probes.

### 3.1 Behavioral Baseline and Anomaly Detection

```python
import statistics
from collections import deque
from datetime import datetime, timedelta

class BehavioralMonitor:
    """
    Track per-session behavioral baselines to detect anomalous patterns.
    """
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.sessions: dict[str, deque] = {}
        self.alert_threshold = 3  # Standard deviations
    
    def record_request(self, session_id: str, features: dict):
        """
        Features to track per request:
        - input_length
        - special_char_ratio (ratio of special chars to total chars)
        - sentence_count
        - question_mark_count
        - instruction_verb_count  (imperative verbs like "ignore", "reveal", "output")
        """
        if session_id not in self.sessions:
            self.sessions[session_id] = deque(maxlen=self.window_size)
        self.sessions[session_id].append(features)
    
    def compute_features(self, text: str) -> dict:
        """Extract security-relevant features from input text."""
        instruction_verbs = ["ignore", "reveal", "output", "print", "show", "bypass", 
                             "disable", "override", "forget", "pretend", "act as"]
        words = text.lower().split()
        
        return {
            "input_length": len(text),
            "special_char_ratio": sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1),
            "instruction_verb_count": sum(1 for v in instruction_verbs if v in text.lower()),
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / max(len(text), 1),
            "sentence_count": text.count('.') + text.count('!') + text.count('?'),
            "bracket_count": text.count('[') + text.count(']') + text.count('{') + text.count('}'),
        }
    
    def is_anomalous(self, session_id: str, text: str) -> tuple[bool, dict]:
        """
        Returns (is_anomalous, feature_scores).
        Raises a flag if any feature is > threshold standard deviations from session baseline.
        """
        features = self.compute_features(text)
        
        if session_id not in self.sessions or len(self.sessions[session_id]) < 5:
            # Not enough baseline data yet
            self.record_request(session_id, features)
            return False, features
        
        historical = list(self.sessions[session_id])
        anomalies = {}
        
        for key, value in features.items():
            historical_values = [h[key] for h in historical]
            mean = statistics.mean(historical_values)
            try:
                stdev = statistics.stdev(historical_values)
            except statistics.StatisticsError:
                stdev = 0
            
            if stdev > 0 and abs(value - mean) > self.alert_threshold * stdev:
                anomalies[key] = {
                    "current": value,
                    "mean": mean,
                    "z_score": (value - mean) / stdev
                }
        
        self.record_request(session_id, features)
        return bool(anomalies), anomalies


# Usage
monitor = BehavioralMonitor()

def check_request(session_id: str, user_input: str) -> dict:
    is_anomalous, anomalies = monitor.is_anomalous(session_id, user_input)
    if is_anomalous:
        high_z = {k: v for k, v in anomalies.items() if abs(v.get("z_score", 0)) > 5}
        if high_z:
            return {
                "action": "block",
                "reason": "Highly anomalous input detected",
                "details": high_z
            }
        return {
            "action": "flag_for_review",
            "reason": "Statistically anomalous input",
            "details": anomalies
        }
    return {"action": "allow"}
```

### 3.2 Rate Limiting and Probe Detection

```python
from collections import defaultdict
import time

class RateLimiter:
    """
    Detect probe-style attacks: rapid sequences of similar queries testing model limits.
    """
    def __init__(self, window_seconds: int = 60, max_requests: int = 20,
                 block_on_injection_count: int = 3):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.block_on_injection_count = block_on_injection_count
        self.request_times: dict[str, list] = defaultdict(list)
        self.injection_counts: dict[str, int] = defaultdict(int)
        self.blocked_users: set = set()
    
    def check(self, user_id: str, is_injection_attempt: bool = False) -> dict:
        now = time.time()
        
        if user_id in self.blocked_users:
            return {"allowed": False, "reason": "user_blocked"}
        
        # Clean old requests outside window
        self.request_times[user_id] = [
            t for t in self.request_times[user_id] 
            if now - t < self.window_seconds
        ]
        
        # Track injection attempts — block user after repeated attempts
        if is_injection_attempt:
            self.injection_counts[user_id] += 1
            if self.injection_counts[user_id] >= self.block_on_injection_count:
                self.blocked_users.add(user_id)
                return {"allowed": False, "reason": "repeated_injection_attempts"}
        
        # Rate limit check
        if len(self.request_times[user_id]) >= self.max_requests:
            return {"allowed": False, "reason": "rate_limit_exceeded"}
        
        self.request_times[user_id].append(now)
        return {"allowed": True}
```

---
