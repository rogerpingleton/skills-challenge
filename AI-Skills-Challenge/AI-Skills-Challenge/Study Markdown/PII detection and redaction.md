# PII detection and redaction

## 6. PII Detection and Redaction {#pii-detection-and-redaction}

### 6.1 Using Microsoft Presidio for PII Detection

Microsoft Presidio is the leading open-source library for PII detection and anonymization in Python.

```python
# Install: pip install presidio-analyzer presidio-anonymizer
# Also: python -m spacy download en_core_web_lg

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

def setup_pii_pipeline():
    """Initialize Presidio analyzer and anonymizer."""
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer

def detect_pii(text: str, analyzer: AnalyzerEngine) -> list:
    """
    Detect PII entities in text.
    Returns list of found entities with type, score, and position.
    """
    results = analyzer.analyze(
        text=text,
        entities=[
            "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
            "US_SSN", "US_PASSPORT", "IBAN_CODE", "IP_ADDRESS",
            "LOCATION", "DATE_TIME", "NRP",  # Nationality, religion, political group
            "MEDICAL_LICENSE", "URL"
        ],
        language="en"
    )
    return results

def redact_pii(text: str, analyzer: AnalyzerEngine, 
               anonymizer: AnonymizerEngine,
               strategy: str = "replace") -> dict:
    """
    Detect and redact PII from text.
    
    Strategies:
    - "replace": Replace with entity type label [PERSON], [EMAIL_ADDRESS]
    - "hash": Replace with SHA-256 hash (reversible with key)
    - "mask": Replace with asterisks
    - "fake": Replace with synthetic data (requires faker)
    """
    results = detect_pii(text, analyzer)
    
    if not results:
        return {"redacted_text": text, "entities_found": [], "pii_detected": False}
    
    operators = {}
    if strategy == "replace":
        operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"})}
    elif strategy == "mask":
        operators = {"DEFAULT": OperatorConfig("mask", {"masking_char": "*", "chars_to_mask": 100, "from_end": False})}
    elif strategy == "hash":
        operators = {"DEFAULT": OperatorConfig("hash", {"hash_type": "sha256"})}
    
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators
    )
    
    return {
        "redacted_text": anonymized.text,
        "entities_found": [
            {"type": r.entity_type, "score": r.score, "start": r.start, "end": r.end}
            for r in results
        ],
        "pii_detected": True
    }


# Complete PII-aware LLM pipeline
def pii_safe_llm_call(client, user_input: str, system_prompt: str) -> dict:
    """
    Full PII-safe pipeline:
    1. Detect PII in input
    2. Redact before sending to LLM
    3. Detect PII in output
    4. Return clean response
    """
    analyzer, anonymizer = setup_pii_pipeline()
    
    # Step 1: Redact PII from input
    input_result = redact_pii(user_input, analyzer, anonymizer, strategy="replace")
    
    if input_result["pii_detected"]:
        print(f"[PII DETECTED IN INPUT] Entities: {[e['type'] for e in input_result['entities_found']]}")
        cleaned_input = input_result["redacted_text"]
    else:
        cleaned_input = user_input
    
    # Step 2: LLM call with cleaned input
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1000,
        system=system_prompt,
        messages=[{"role": "user", "content": cleaned_input}]
    )
    raw_output = response.content[0].text
    
    # Step 3: Check output for PII leakage
    output_result = redact_pii(raw_output, analyzer, anonymizer, strategy="replace")
    
    if output_result["pii_detected"]:
        print(f"[PII DETECTED IN OUTPUT - REDACTED] Entities: {[e['type'] for e in output_result['entities_found']]}")
        final_output = output_result["redacted_text"]
    else:
        final_output = raw_output
    
    return {
        "response": final_output,
        "input_pii_detected": input_result["pii_detected"],
        "output_pii_detected": output_result["pii_detected"],
        "input_entities": input_result.get("entities_found", []),
        "output_entities": output_result.get("entities_found", [])
    }


# Example usage
if __name__ == "__main__":
    sample_text = """
    Hi, my name is John Smith. My email is john.smith@example.com 
    and my SSN is 123-45-6789. Please review my account.
    """
    analyzer, anonymizer = setup_pii_pipeline()
    result = redact_pii(sample_text, analyzer, anonymizer)
    print("Original:", sample_text)
    print("Redacted:", result["redacted_text"])
    print("Entities:", result["entities_found"])
```

### 6.2 Custom PII Patterns for Domain-Specific Data

```python
from presidio_analyzer import PatternRecognizer, Pattern

def add_custom_recognizers(analyzer: AnalyzerEngine):
    """Add domain-specific PII recognizers."""
    
    # Example: Employee ID recognizer
    employee_id_pattern = Pattern(
        name="employee_id_pattern",
        regex=r"\bEMP-[0-9]{6}\b",
        score=0.9
    )
    employee_recognizer = PatternRecognizer(
        supported_entity="EMPLOYEE_ID",
        patterns=[employee_id_pattern]
    )
    analyzer.registry.add_recognizer(employee_recognizer)
    
    # Example: Medical Record Number
    mrn_pattern = Pattern(
        name="mrn_pattern",
        regex=r"\bMRN[:\s]?[0-9]{8,12}\b",
        score=0.85
    )
    mrn_recognizer = PatternRecognizer(
        supported_entity="MEDICAL_RECORD_NUMBER",
        patterns=[mrn_pattern]
    )
    analyzer.registry.add_recognizer(mrn_recognizer)
    
    return analyzer
```
