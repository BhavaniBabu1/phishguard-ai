"""
utils.py - Feature Extraction Module
======================================
Extracts numerical features from URLs for ML classification.
Each feature captures a known signal used by phishing sites.
"""

import re
import urllib.parse
from typing import Dict, List


# ─── Suspicious keyword lists ──────────────────────────────────────────────

PHISHING_KEYWORDS = [
    "login", "signin", "sign-in", "secure", "security", "verify",
    "verification", "account", "update", "confirm", "banking",
    "password", "credential", "wallet", "alert", "suspended",
    "locked", "unusual", "activity", "recover", "support",
    "billing", "payment", "invoice", "refund", "prize", "free",
    "winner", "claim", "bonus", "gift", "lucky", "congratulations",
    "urgent", "immediately", "validate", "authorize", "access",
]

SUSPICIOUS_TLDS = [
    ".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".club",
    ".info", ".biz", ".win", ".loan", ".click", ".link", ".work",
    ".online", ".website", ".site", ".tech", ".space",
]

TRUSTED_BRANDS = [
    "paypal", "amazon", "google", "facebook", "apple", "microsoft",
    "netflix", "ebay", "instagram", "twitter", "linkedin", "bank",
    "chase", "wellsfargo", "citibank", "hsbc", "barclays",
    "steam", "roblox", "discord", "binance", "coinbase",
]


# ─── Individual feature extractors ─────────────────────────────────────────

def get_url_length(url: str) -> int:
    """Phishing URLs are often unusually long to obscure the real domain."""
    return len(url)


def has_at_symbol(url: str) -> int:
    """'@' in URL forces browser to ignore everything before it (attacker trick)."""
    return 1 if "@" in url else 0


def has_ip_address(url: str) -> int:
    """Direct IP instead of domain name is a strong phishing signal."""
    ip_pattern = r"(\d{1,3}\.){3}\d{1,3}"
    return 1 if re.search(ip_pattern, url) else 0


def get_subdomain_count(url: str) -> int:
    """Phishing pages stack subdomains to mimic legit sites (e.g. paypal.com.evil.tk)."""
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        parts = hostname.split(".")
        # subtract root domain + TLD
        return max(0, len(parts) - 2)
    except Exception:
        return 0


def uses_https(url: str) -> int:
    """HTTPS is expected for legit sites; absence is suspicious."""
    return 1 if url.lower().startswith("https://") else 0


def has_suspicious_keywords(url: str) -> int:
    """Check if URL contains phishing-associated keywords."""
    url_lower = url.lower()
    return 1 if any(kw in url_lower for kw in PHISHING_KEYWORDS) else 0


def count_suspicious_keywords(url: str) -> int:
    """Count how many phishing keywords appear (more = worse)."""
    url_lower = url.lower()
    return sum(1 for kw in PHISHING_KEYWORDS if kw in url_lower)


def has_suspicious_tld(url: str) -> int:
    """Free / abused TLDs are commonly used in phishing campaigns."""
    url_lower = url.lower()
    return 1 if any(url_lower.endswith(tld) or (tld + "/") in url_lower
                    for tld in SUSPICIOUS_TLDS) else 0


def has_brand_impersonation(url: str) -> int:
    """Legitimate brand name in URL but not as the primary domain."""
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        parts = hostname.split(".")
        # Primary domain is second-to-last part
        primary_domain = parts[-2] if len(parts) >= 2 else hostname
        url_lower = url.lower()
        for brand in TRUSTED_BRANDS:
            if brand in url_lower and brand not in primary_domain:
                return 1
        return 0
    except Exception:
        return 0


def get_special_char_count(url: str) -> int:
    """Count special chars ('-', '_', '~', '%') — phishing URLs often overuse them."""
    return sum(url.count(c) for c in ["-", "_", "~", "%"])


def has_double_slash(url: str) -> int:
    """Double slash after protocol part can indicate URL redirect tricks."""
    return 1 if "//" in url[8:] else 0


def get_path_depth(url: str) -> int:
    """Deep paths are common in phishing URLs to look like legit portals."""
    try:
        parsed = urllib.parse.urlparse(url)
        return len([p for p in parsed.path.split("/") if p])
    except Exception:
        return 0


def has_query_params(url: str) -> int:
    """Presence of query parameters (often used to pass victim data)."""
    try:
        parsed = urllib.parse.urlparse(url)
        return 1 if parsed.query else 0
    except Exception:
        return 0


def uses_url_shortener(url: str) -> int:
    """Known URL shorteners used to mask phishing destinations."""
    shorteners = ["bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly",
                  "shorte.st", "clck.ru", "is.gd", "buff.ly", "adf.ly"]
    return 1 if any(s in url.lower() for s in shorteners) else 0


def get_digit_ratio(url: str) -> float:
    """Ratio of digits in domain — legit domains rarely have many digits."""
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        if not hostname:
            return 0.0
        digits = sum(c.isdigit() for c in hostname)
        return round(digits / len(hostname), 4)
    except Exception:
        return 0.0


def get_domain_length(url: str) -> int:
    """Short or very long domain names can both be suspicious."""
    try:
        parsed = urllib.parse.urlparse(url)
        hostname = parsed.hostname or ""
        parts = hostname.split(".")
        primary = parts[-2] if len(parts) >= 2 else hostname
        return len(primary)
    except Exception:
        return 0


# ─── Master feature extractor ───────────────────────────────────────────────

FEATURE_NAMES = [
    "url_length",
    "has_at_symbol",
    "has_ip_address",
    "subdomain_count",
    "uses_https",
    "has_suspicious_keywords",
    "suspicious_keyword_count",
    "has_suspicious_tld",
    "has_brand_impersonation",
    "special_char_count",
    "has_double_slash",
    "path_depth",
    "has_query_params",
    "uses_url_shortener",
    "digit_ratio",
    "domain_length",
]


def extract_features(url: str) -> Dict:
    """
    Extract all features from a URL and return as a labelled dict.
    This is used both for training (batch) and inference (single URL).
    """
    return {
        "url_length": get_url_length(url),
        "has_at_symbol": has_at_symbol(url),
        "has_ip_address": has_ip_address(url),
        "subdomain_count": get_subdomain_count(url),
        "uses_https": uses_https(url),
        "has_suspicious_keywords": has_suspicious_keywords(url),
        "suspicious_keyword_count": count_suspicious_keywords(url),
        "has_suspicious_tld": has_suspicious_tld(url),
        "has_brand_impersonation": has_brand_impersonation(url),
        "special_char_count": get_special_char_count(url),
        "has_double_slash": has_double_slash(url),
        "path_depth": get_path_depth(url),
        "has_query_params": has_query_params(url),
        "uses_url_shortener": uses_url_shortener(url),
        "digit_ratio": get_digit_ratio(url),
        "domain_length": get_domain_length(url),
    }


def extract_feature_vector(url: str) -> List:
    """Return ordered list of feature values (for model prediction)."""
    features = extract_features(url)
    return [features[name] for name in FEATURE_NAMES]


def get_feature_display(url: str) -> List[Dict]:
    """
    Return features formatted for frontend display with human-readable
    labels, values, and risk assessment for each feature.
    """
    features = extract_features(url)

    display = [
        {
            "name": "URL Length",
            "value": features["url_length"],
            "risk": "high" if features["url_length"] > 75 else
                    "medium" if features["url_length"] > 54 else "low",
            "description": "Long URLs often mask malicious destinations",
        },
        {
            "name": "@ Symbol",
            "value": "Yes" if features["has_at_symbol"] else "No",
            "risk": "high" if features["has_at_symbol"] else "low",
            "description": "Browser ignores everything before @ in a URL",
        },
        {
            "name": "IP Address",
            "value": "Detected" if features["has_ip_address"] else "None",
            "risk": "high" if features["has_ip_address"] else "low",
            "description": "Legitimate sites use domain names, not raw IPs",
        },
        {
            "name": "Subdomains",
            "value": features["subdomain_count"],
            "risk": "high" if features["subdomain_count"] > 3 else
                    "medium" if features["subdomain_count"] > 1 else "low",
            "description": "Stacked subdomains mimic trusted domains",
        },
        {
            "name": "HTTPS",
            "value": "Secure" if features["uses_https"] else "Insecure",
            "risk": "low" if features["uses_https"] else "medium",
            "description": "Absence of HTTPS is a warning sign",
        },
        {
            "name": "Suspicious Keywords",
            "value": features["suspicious_keyword_count"],
            "risk": "high" if features["suspicious_keyword_count"] > 2 else
                    "medium" if features["suspicious_keyword_count"] > 0 else "low",
            "description": "Words like 'login', 'verify', 'secure' are phishing bait",
        },
        {
            "name": "Suspicious TLD",
            "value": "Yes" if features["has_suspicious_tld"] else "No",
            "risk": "high" if features["has_suspicious_tld"] else "low",
            "description": "Free/abused TLDs (.tk, .xyz) are common in phishing",
        },
        {
            "name": "Brand Impersonation",
            "value": "Detected" if features["has_brand_impersonation"] else "None",
            "risk": "high" if features["has_brand_impersonation"] else "low",
            "description": "Brand name in URL but not as the real domain",
        },
        {
            "name": "URL Shortener",
            "value": "Yes" if features["uses_url_shortener"] else "No",
            "risk": "medium" if features["uses_url_shortener"] else "low",
            "description": "Shorteners hide the true destination",
        },
        {
            "name": "Digit Ratio",
            "value": f"{features['digit_ratio']:.1%}",
            "risk": "medium" if features["digit_ratio"] > 0.2 else "low",
            "description": "High digit density in domain is unusual",
        },
    ]

    return display