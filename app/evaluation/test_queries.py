"""
Test Queries for Evaluation

Test queries with ground truth for Electric Inc. RAG system.
Uses content-based relevance matching (source files + keywords) so the same
ground truth works across FAISS and pgvector systems.

Source filenames verified against FAISS index metadata:
  - HRPolicies_ElectricInc.pdf
  - Electric_Inc_Onboarding.pdf
  - Electric_Inc_values.pdf
  - Report_25.pdf
  - New_Products.pdf
  - Marketing_Electric_Inc.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentTestQuery:
    """Test query with content-based ground truth.

    A retrieved chunk is considered relevant if:
      1. Its metadata ``source`` contains one of ``relevant_sources``, AND
      2. Its ``page_content`` contains at least one of ``relevant_keywords``
         (case-insensitive substring match).
    """
    id: str
    query: str
    relevant_sources: list[str]
    relevant_keywords: list[str]
    relevance_weight: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


TEST_QUERIES: list[ContentTestQuery] = [
    # ── HR Policy ──────────────────────────────────────────────────────────
    ContentTestQuery(
        id="hr-001",
        query="What is the vacation policy at Electric Inc?",
        relevant_sources=["HRPolicies_ElectricInc.pdf"],
        relevant_keywords=["holiday", "leave", "vacation", "carryover", "payout"],
        metadata={"category": "hr-policy", "difficulty": "easy"},
    ),
    ContentTestQuery(
        id="hr-002",
        query="How many sick days do employees get per year?",
        relevant_sources=["HRPolicies_ElectricInc.pdf"],
        relevant_keywords=["sick", "10 days", "sick leave"],
        metadata={"category": "hr-policy", "difficulty": "easy"},
    ),
    ContentTestQuery(
        id="hr-003",
        query="Can I carry over unused vacation days to next year?",
        relevant_sources=["HRPolicies_ElectricInc.pdf"],
        relevant_keywords=["carryover", "carry over", "5 days", "payout", "termination"],
        metadata={"category": "hr-policy", "difficulty": "medium"},
    ),

    # ── Security & Data Protection ─────────────────────────────────────────
    ContentTestQuery(
        id="sec-001",
        query="What is the data security policy at Electric Inc?",
        relevant_sources=["HRPolicies_ElectricInc.pdf"],
        relevant_keywords=["data security", "VPN", "home network", "encryption", "security"],
        metadata={"category": "security", "difficulty": "easy"},
    ),
    ContentTestQuery(
        id="sec-002",
        query="How do I report a security incident?",
        relevant_sources=["HRPolicies_ElectricInc.pdf"],
        relevant_keywords=["incident", "report", "2 hours", "IT Security", "SIRT"],
        metadata={"category": "security", "difficulty": "medium"},
    ),
    ContentTestQuery(
        id="sec-003",
        query="What are the confidentiality requirements for employees?",
        relevant_sources=["HRPolicies_ElectricInc.pdf"],
        relevant_keywords=["confidentiality", "digital documents", "violation", "training", "compliance"],
        metadata={"category": "security", "difficulty": "medium"},
    ),

    # ── Onboarding ─────────────────────────────────────────────────────────
    ContentTestQuery(
        id="onb-001",
        query="What do I need to do on my first day at Electric Inc?",
        relevant_sources=["Electric_Inc_Onboarding.pdf"],
        relevant_keywords=["first day", "welcome", "orientation", "essentials"],
        metadata={"category": "onboarding", "difficulty": "easy"},
    ),
    ContentTestQuery(
        id="onb-002",
        query="How do I contact IT support?",
        relevant_sources=["Electric_Inc_Onboarding.pdf"],
        relevant_keywords=["IT support", "helpdesk", "help desk", "contact", "IT help"],
        metadata={"category": "onboarding", "difficulty": "easy"},
    ),

    # ── Company Values & Mission ───────────────────────────────────────────
    ContentTestQuery(
        id="val-001",
        query="What is the mission of Electric Inc?",
        relevant_sources=["Electric_Inc_Onboarding.pdf", "Electric_Inc_values.pdf"],
        relevant_keywords=["mission", "vision", "purpose"],
        metadata={"category": "company-values", "difficulty": "easy"},
    ),
    ContentTestQuery(
        id="val-002",
        query="What are the sustainability goals of Electric Inc?",
        relevant_sources=["Electric_Inc_values.pdf", "Report_25.pdf"],
        relevant_keywords=["sustainability", "carbon", "environmental", "2028", "reduction"],
        metadata={"category": "company-values", "difficulty": "medium"},
    ),
    ContentTestQuery(
        id="val-003",
        query="What are the core values of the company?",
        relevant_sources=["Electric_Inc_values.pdf"],
        relevant_keywords=["values", "integrity", "innovation", "quality", "sustainability"],
        metadata={"category": "company-values", "difficulty": "easy"},
    ),

    # ── Products ───────────────────────────────────────────────────────────
    ContentTestQuery(
        id="prod-001",
        query="What new products did Electric Inc launch in 2025?",
        relevant_sources=["Report_25.pdf", "New_Products.pdf"],
        relevant_keywords=["EcoCharge", "Grid", "2025", "new product", "launch", "AI-Driven"],
        metadata={"category": "products", "difficulty": "medium"},
    ),
    ContentTestQuery(
        id="prod-002",
        query="Tell me about the EcoCharge Pro product",
        relevant_sources=["Report_25.pdf", "New_Products.pdf"],
        relevant_keywords=["EcoCharge Pro", "EV charger", "smart"],
        metadata={"category": "products", "difficulty": "medium"},
    ),

    # ── Marketing ──────────────────────────────────────────────────────────
    ContentTestQuery(
        id="mkt-001",
        query="What is the brand voice for Electric Inc marketing?",
        relevant_sources=["Marketing_Electric_Inc.pdf"],
        relevant_keywords=["brand voice", "professional", "innovative", "tone"],
        metadata={"category": "marketing", "difficulty": "medium"},
    ),
    ContentTestQuery(
        id="mkt-002",
        query="What digital marketing channels does Electric Inc use?",
        relevant_sources=["Marketing_Electric_Inc.pdf"],
        relevant_keywords=["email", "social media", "website", "digital", "channel"],
        metadata={"category": "marketing", "difficulty": "medium"},
    ),

    # ── Strategy ───────────────────────────────────────────────────────────
    ContentTestQuery(
        id="str-001",
        query="What is the company strategy for 2025?",
        relevant_sources=["Report_25.pdf"],
        relevant_keywords=["strategy", "2025", "growth", "expansion", "roadmap"],
        metadata={"category": "strategy", "difficulty": "medium"},
    ),

    # ── Benefits & Perks ──────────────────────────────────────────────────
    ContentTestQuery(
        id="ben-001",
        query="What wellness programs are available for employees?",
        relevant_sources=["HRPolicies_ElectricInc.pdf", "Electric_Inc_Onboarding.pdf"],
        relevant_keywords=["wellness", "yoga", "meditation", "gym", "stipend"],
        metadata={"category": "hr-policy", "difficulty": "medium"},
    ),
    ContentTestQuery(
        id="ben-002",
        query="Does the company organize social events?",
        relevant_sources=["HRPolicies_ElectricInc.pdf", "Electric_Inc_Onboarding.pdf"],
        relevant_keywords=["social event", "offsite", "happy hour", "holiday", "celebration", "team"],
        metadata={"category": "hr-policy", "difficulty": "medium"},
    ),
]


def get_test_queries(
    category: str | None = None,
    difficulty: str | None = None,
    limit: int | None = None,
) -> list[ContentTestQuery]:
    """Return test queries, optionally filtered by category/difficulty."""
    filtered = list(TEST_QUERIES)
    if category:
        filtered = [q for q in filtered if q.metadata.get("category") == category]
    if difficulty:
        filtered = [q for q in filtered if q.metadata.get("difficulty") == difficulty]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered
