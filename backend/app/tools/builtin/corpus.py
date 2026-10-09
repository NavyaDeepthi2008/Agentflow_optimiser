"""A small, fictional knowledge base used by the mock search and read tools.

"Nimbus Labs" and everything about it is invented, so expected answers in the
evaluation dataset are known exactly and never depend on the real internet.
"""

from __future__ import annotations

DOCUMENTS: dict[str, dict[str, str]] = {
    "kb-001": {
        "title": "Nimbus Labs company overview",
        "body": (
            "Nimbus Labs is a fictional software company founded in 2019. Its headquarters "
            "is in Pune and it has 140 employees. Nimbus Labs sells two products: Orbit, a "
            "team scheduling application, and Keystone, a data backup appliance."
        ),
    },
    "kb-002": {
        "title": "Orbit product sheet",
        "body": (
            "Orbit is a team scheduling application launched in 2021. The current version "
            "is 4.2. Orbit costs 29 USD per user per month on the monthly plan and 290 USD "
            "per user per year on the annual plan."
        ),
    },
    "kb-003": {
        "title": "Keystone product sheet",
        "body": (
            "Keystone is a data backup appliance launched in 2023. The current firmware "
            "version is 1.8. One Keystone unit costs 1200 USD and includes 2 TB of storage."
        ),
    },
    "kb-004": {
        "title": "Refund policy",
        "body": (
            "Customers may request a full refund within 30 days of purchase. After 30 days, "
            "annual plans are refunded pro rata for the unused months. Monthly plans are not "
            "refunded after 30 days. Refund requests are handled by the billing team."
        ),
    },
    "kb-005": {
        "title": "Support hours",
        "body": (
            "Standard support is available Monday to Friday, 09:00 to 18:00 IST. Enterprise "
            "customers receive priority support 24 hours a day, 7 days a week."
        ),
    },
    "kb-006": {
        "title": "Travel policy",
        "body": (
            "Employees book economy class for flights shorter than 6 hours. The daily meal "
            "allowance during business travel is 1500 INR. Hotel bookings need manager "
            "approval."
        ),
    },
    "kb-007": {
        "title": "Security incident procedure",
        "body": (
            "Any suspected security incident must be reported to the security team within "
            "1 hour. Incidents are classified as severity 1 (critical), severity 2 (major) "
            "or severity 3 (minor)."
        ),
    },
    "kb-008": {
        "title": "Orbit 4.2 release notes",
        "body": (
            "Orbit 4.2 was released on 2026-03-14. It added two-way calendar sync and a "
            "dark theme. Orbit 4.3 is planned for the fourth quarter of 2026."
        ),
    },
}

# Starting values for the mock price tool (fictional instruments, not real tickers).
BASE_PRICES: dict[str, float] = {
    "NMB-X": 120.00,
    "ORB-X": 45.50,
    "KST-X": 310.25,
}

# Starting records for the mock record tools.
SEED_RECORDS: dict[str, dict[str, object]] = {
    "REC-1001": {
        "id": "REC-1001",
        "title": "Orbit calendar sync fails for some users",
        "status": "open",
        "owner": "team:orbit",
        "note": "",
        "version": 1,
    },
    "REC-1002": {
        "id": "REC-1002",
        "title": "Keystone firmware 1.8 rollout",
        "status": "in_progress",
        "owner": "team:keystone",
        "note": "",
        "version": 1,
    },
    "REC-1003": {
        "id": "REC-1003",
        "title": "Update refund policy wording",
        "status": "resolved",
        "owner": "team:billing",
        "note": "",
        "version": 1,
    },
}
