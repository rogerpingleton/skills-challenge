
# Legal compliance (GDPR, copyright implications)

## 7. Legal Compliance in AI Engineering {#legal-compliance}

### 7.1 GDPR {#gdpr}

The EU General Data Protection Regulation imposes strict requirements on any system that processes personal data of EU residents — including LLM applications.

**Key GDPR Articles for AI Engineers:**

| Article | Requirement | AI Engineering Implication |
|---------|-------------|---------------------------|
| Art. 5 | Data minimization | Don't feed LLMs more PII than necessary |
| Art. 17 | Right to erasure | Must support deletion from training data and vector DBs |
| Art. 22 | Automated decision-making | Significant AI-driven decisions require human oversight |
| Art. 25 | Privacy by design | Build PII controls into the LLM pipeline from the start |
| Art. 32 | Security of processing | Implement technical controls (encryption, access control) |
| Art. 35 | DPIA required | Conduct Data Protection Impact Assessment for high-risk AI |

**GDPR Compliance Checklist for LLM Applications:**

```python
# gdpr_compliance.py

GDPR_CHECKLIST = {
    "data_minimization": {
        "description": "Only process PII that is strictly necessary",
        "implementation": [
            "Redact PII before sending to external LLM APIs",
            "Use pseudonymization tokens for user identification",
            "Implement purpose limitation in system prompts",
        ]
    },
    "right_to_erasure": {
        "description": "Be able to delete a user's data from all LLM-related stores",
        "implementation": [
            "Maintain an inventory of all vector DB / RAG stores containing user data",
            "Implement deletion pipelines for user data from fine-tuning datasets",
            "Log all LLM interactions with user ID linkage for traceability",
        ]
    },
    "privacy_by_design": {
        "description": "PII protection built in from the start",
        "implementation": [
            "Use Presidio or similar for real-time PII detection/redaction",
            "Apply differential privacy to fine-tuning workflows",
            "Configure session isolation to prevent cross-user data leakage",
        ]
    },
    "data_retention": {
        "description": "Don't retain data longer than necessary",
        "implementation": [
            "Implement automatic conversation log expiry",
            "Document and enforce retention policies for LLM inputs/outputs",
            "Avoid using conversation history for training without explicit consent",
        ]
    },
    "transparency": {
        "description": "Users must know AI is being used",
        "implementation": [
            "Disclose AI usage in user-facing applications",
            "Document which LLM models and providers are used",
            "Provide users with rights to contest automated decisions",
        ]
    }
}

def generate_gdpr_audit_report(application_name: str) -> str:
    """Generate a basic GDPR compliance self-audit template."""
    lines = [f"# GDPR Compliance Self-Audit: {application_name}\n"]
    for area, details in GDPR_CHECKLIST.items():
        lines.append(f"## {area.replace('_', ' ').title()}")
        lines.append(f"**Requirement:** {details['description']}\n")
        lines.append("**Implementation checklist:**")
        for item in details["implementation"]:
            lines.append(f"- [ ] {item}")
        lines.append("")
    return "\n".join(lines)
```

### 7.2 HIPAA and CCPA {#hipaa-and-ccpa}

**HIPAA (Healthcare):** Any LLM processing Protected Health Information (PHI) must:
- Implement access controls and audit logging
- Apply PHI redaction before processing with external models
- Ensure Business Associate Agreements (BAAs) are in place with AI vendors
- Never retain PHI in LLM context windows beyond immediate use

**CCPA (California):** Consumer rights include:
- Right to know what personal information is collected by AI
- Right to opt out of "sale" of personal information (which may include model training)
- Right to deletion

```python
# Simple PHI detection additions for Presidio
PHI_ENTITIES = [
    "PERSON",
    "DATE_TIME",          # Dates of birth, service dates
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "LOCATION",           # Addresses, geographic data
    "US_SSN",
    "MEDICAL_LICENSE",
    # Custom: add diagnosis codes, procedure codes via custom recognizers
]

def is_hipaa_compliant_for_llm(text: str, analyzer: AnalyzerEngine) -> tuple[bool, list]:
    """
    Check if text is safe to send to an external LLM under HIPAA.
    Returns (is_safe, detected_phi).
    """
    results = analyzer.analyze(text=text, entities=PHI_ENTITIES, language="en")
    high_confidence = [r for r in results if r.score >= 0.7]
    return len(high_confidence) == 0, high_confidence
```

### 7.3 Copyright Implications {#copyright-implications}

As of 2025–2026, copyright law and AI is actively being shaped by litigation and regulation.

**Key 2025 Developments:**

- In **Bartz v. Anthropic PBC** (June 2025), a federal court ruled that training LLMs on lawfully acquired books is potentially "transformative" fair use, but retaining pirated copies as part of a training library was **not** protected. This means: provenance of training data matters enormously.
- In **Thomson Reuters v. ROSS Intelligence** (February 2025), a court rejected a fair use defense where training data was used to build a directly competing product. The competitive market effect was a decisive factor.
- The **U.S. Copyright Office Part 3 Report** (May 2025) concluded that AI-generated content is **not copyrightable** unless a human has determined sufficient expressive elements. Simply writing prompts is not enough to claim authorship.
- The **EU AI Act** requires AI developers to maintain detailed records of training data and disclose copyrighted works used in training.

**Practical Guidelines for AI Engineers:**

```python
# training_data_compliance.py

class TrainingDataRecord:
    """Track provenance and licensing of training data for copyright compliance."""
    
    def __init__(self):
        self.records = []
    
    def add_dataset(self, name: str, source: str, license_type: str,
                    acquisition_method: str, has_robots_txt_opt_in: bool = None,
                    contains_copyrighted_works: bool = False,
                    licensed_or_public_domain: bool = False):
        """
        Document each training dataset for audit purposes.
        
        license_type: "public_domain", "cc0", "cc_by", "licensed", "scraped_unknown",
                      "proprietary_consented"
        acquisition_method: "licensed", "public_domain", "web_scrape", "synthetic",
                            "user_contributed_consented"
        """
        risk_score = self._calculate_risk(license_type, acquisition_method,
                                          contains_copyrighted_works, licensed_or_public_domain)
        
        self.records.append({
            "name": name,
            "source": source,
            "license_type": license_type,
            "acquisition_method": acquisition_method,
            "has_robots_txt_opt_in": has_robots_txt_opt_in,
            "contains_copyrighted_works": contains_copyrighted_works,
            "licensed_or_public_domain": licensed_or_public_domain,
            "copyright_risk_score": risk_score,
            "notes": self._get_risk_notes(risk_score)
        })
    
    def _calculate_risk(self, license_type, acquisition_method,
                        contains_copyrighted, is_licensed) -> str:
        if license_type in ("public_domain", "cc0") or is_licensed:
            return "LOW"
        if acquisition_method == "web_scrape" and contains_copyrighted:
            return "HIGH"
        if license_type == "scraped_unknown":
            return "MEDIUM_HIGH"
        return "MEDIUM"
    
    def _get_risk_notes(self, risk: str) -> str:
        notes = {
            "LOW": "Proceed. Ensure license terms are documented.",
            "MEDIUM": "Review license terms. Consult legal if competitive market overlap.",
            "MEDIUM_HIGH": "Consult legal. Consider licensing or replacing with public domain data.",
            "HIGH": "Do not use without explicit legal clearance. High infringement risk per ROSS ruling."
        }
        return notes.get(risk, "Unknown risk level.")
    
    def get_high_risk_datasets(self) -> list:
        return [r for r in self.records if r["copyright_risk_score"] in ("HIGH", "MEDIUM_HIGH")]
    
    def export_audit_report(self) -> str:
        """Export EU AI Act-compliant training data disclosure."""
        import json
        return json.dumps(self.records, indent=2)
```

**Copyright Best Practices:**
- Prefer licensed datasets, public domain content, or synthetic data for training
- Audit outputs using plagiarism detectors (e.g., CopyLeaks, iThenticate) before publishing
- Do not prompt models to reproduce copyrighted works (books, articles, song lyrics)
- If your model may output material "substantially similar" to copyrighted works, implement output similarity scanning
- Maintain detailed training data provenance records as required by the EU AI Act

### 7.4 EU AI Act {#eu-ai-act}

The EU AI Act (effective 2024–2026) classifies AI systems by risk level:

| Risk Level | Examples | Obligations |
|------------|----------|-------------|
| Unacceptable | Social scoring, real-time biometric surveillance | **Prohibited** |
| High | Healthcare AI, HR recruitment, critical infrastructure | Conformity assessment, human oversight, transparency |
| Limited | Chatbots, AI-generated content | Transparency disclosure required |
| Minimal | Spam filters, AI-recommended content | Minimal requirements |

**Key obligations for most AI Engineers:**
- Maintain training data documentation (including copyright provenance)
- Implement human oversight for high-risk decisions
- Ensure transparency (users must know they're interacting with AI)
- Establish conformity assessment processes for high-risk systems
- Register high-risk AI in the EU database

---
