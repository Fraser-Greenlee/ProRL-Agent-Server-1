import math
import re
from typing import Any

REDACTED = "[REDACTED]"

# Ordered from most specific to least specific
SECRET_PATTERNS = [
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}")),
    ("jwt_partial", re.compile(r"eyJ[A-Za-z0-9_-]{15,}")),
    ("db_url", re.compile(r"postgres(?:ql)?://[^:]+:[^@\s]+@[^\s\"'`]+")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai_key", re.compile(r"sk-[A-Za-z0-9]{40,}")),
    ("hf_token", re.compile(r"hf_[A-Za-z0-9]{20,}")),
    ("github_token", re.compile(r"(?:ghp|gho|ghs|ghr)_[A-Za-z0-9]{30,}")),
    ("pypi_token", re.compile(r"pypi-[A-Za-z0-9_-]{50,}")),
    ("npm_token", re.compile(r"npm_[A-Za-z0-9]{30,}")),
    ("aws_key", re.compile(r"(?<![A-Za-z0-9\[])AKIA[0-9A-Z]{16}(?![0-9A-Z\]{}])")),
    ("aws_secret", re.compile(
        r"(?:aws_secret_access_key|secret_key)\s*[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?",
        re.IGNORECASE,
    )),
    ("slack_token", re.compile(r"xox[bpsa]-[A-Za-z0-9-]{20,}")),
    ("discord_webhook", re.compile(
        r"https?://(?:discord\.com|discordapp\.com)/api/webhooks/\d+/[A-Za-z0-9_-]{20,}"
    )),
    ("private_key", re.compile(
        r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
        r"[\s\S]*?"
        r"-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
    )),
    ("cli_token_flag", re.compile(
        r"(?:--|-)(?:access[_-]?token|auth[_-]?token|api[_-]?key|secret|password|token)"
        r"[\s=]+([A-Za-z0-9_/+=.-]{8,})",
        re.IGNORECASE,
    )),
    ("env_secret", re.compile(
        r"(?:SECRET|PASSWORD|TOKEN|API_KEY|AUTH_KEY|ACCESS_KEY|SERVICE_KEY|DB_PASSWORD"
        r"|SUPABASE_KEY|SUPABASE_SERVICE|ANON_KEY|SERVICE_ROLE)"
        r"\s*[=]\s*['\"]?([^\s'\"]{6,})['\"]?",
        re.IGNORECASE,
    )),
    ("generic_secret", re.compile(
        r"""(?:secret[_-]?key|api[_-]?key|api[_-]?secret|access[_-]?token|auth[_-]?token"""
        r"""|service[_-]?role[_-]?key|private[_-]?key)"""
        r"""\s*[=:]\s*['"]([A-Za-z0-9_/+=.-]{20,})['"]""",
        re.IGNORECASE,
    )),
    ("bearer", re.compile(
        r"Bearer\s+(eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,})"
    )),
    ("ip_address", re.compile(
        r"\b(?!127\.0\.0\.)(?!0\.0\.0\.0)(?!255\.255\.)"
        r"(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
    )),
    ("url_token", re.compile(
        r"[?&](?:key|token|secret|password|apikey|api_key|access_token|auth)"
        r"=([A-Za-z0-9_/+=.-]{8,})",
        re.IGNORECASE,
    )),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]{2,}@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("high_entropy", re.compile(r"""['"][A-Za-z0-9_/+=.-]{40,}['"]""")),
]

ALLOWLIST = [
    re.compile(r"noreply@"),
    re.compile(r"@example\.com"),
    re.compile(r"@localhost"),
    re.compile(r"@anthropic\.com"),
    re.compile(r"@github\.com"),
    re.compile(r"@users\.noreply\.github\.com"),
    re.compile(r"AKIA\["),
    re.compile(r"sk-ant-\.\*"),
    re.compile(r"postgres://user:pass@"),
    re.compile(r"postgres://username:password@"),
    re.compile(r"@pytest"),
    re.compile(r"@tasks\."),
    re.compile(r"@mcp\."),
    re.compile(r"@server\."),
    re.compile(r"@app\."),
    re.compile(r"@router\."),
    re.compile(r"192\.168\."),
    re.compile(r"10\.\d+\.\d+\.\d+"),
    re.compile(r"172\.(?:1[6-9]|2\d|3[01])\."),
    re.compile(r"8\.8\.8\.8"),
    re.compile(r"8\.8\.4\.4"),
    re.compile(r"1\.1\.1\.1"),
]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    length = len(s)
    return -sum((count / length) * math.log2(count / length) for count in freq.values())


def _has_mixed_char_types(s: str) -> bool:
    return (
        any(c.isupper() for c in s)
        and any(c.islower() for c in s)
        and any(c.isdigit() for c in s)
    )


def scan_text(text: str) -> list[dict]:
    """Scan text for secrets. Returns list of {type, start, end, match}."""
    if not text:
        return []

    findings = []
    for name, pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            matched_text = match.group(0)
            if any(allow.search(matched_text) for allow in ALLOWLIST):
                continue
            if name == "high_entropy":
                inner = matched_text[1:-1]
                if not _has_mixed_char_types(inner):
                    continue
                if _shannon_entropy(inner) < 3.5:
                    continue
                if inner.count(".") > 2:
                    continue
            findings.append({
                "type": name,
                "start": match.start(),
                "end": match.end(),
                "match": matched_text,
            })
    return findings


def redact_text(text: str) -> tuple[str, int]:
    """Redact secrets in a text string. Returns (redacted_text, count)."""
    if not text:
        return text, 0

    findings = scan_text(text)
    if not findings:
        return text, 0

    findings.sort(key=lambda f: f["start"], reverse=True)
    deduped = []
    for f in findings:
        if not deduped or f["end"] <= deduped[-1]["start"]:
            deduped.append(f)

    result = text
    for f in deduped:
        result = result[: f["start"]] + REDACTED + result[f["end"] :]
    return result, len(deduped)


def redact_value(value: Any) -> tuple[Any, int]:
    """Recursively redact secrets from a string, list, or dict value."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        total = 0
        out = {}
        for k, v in value.items():
            out[k], n = redact_value(v)
            total += n
        return out, total
    if isinstance(value, list):
        total = 0
        out_list = []
        for item in value:
            redacted, n = redact_value(item)
            out_list.append(redacted)
            total += n
        return out_list, total
    return value, 0
