"""Graduation tier model definitions.

Defines the formal tier hierarchy (T0-T3) and the requirements
associated with each tier transition.

Tier overview:
  T0 (Personal)  — single-user tool, no requirements
  T1 (Shared)    — 2+ users, basic docs, no red-tier security issues
  T2 (Team)      — team tool, quality score > 0.7, tests, tech owner
  T3 (Critical)  — production tool, full test coverage, security review, SLA
"""

from __future__ import annotations

from dataclasses import dataclass

from shared.models import ToolTier


@dataclass
class TierDefinition:
    """Definition of a single graduation tier."""

    tier: ToolTier
    label: str
    description: str
    support_level: str
    monitoring: str


# Canonical tier definitions
TIER_DEFINITIONS: dict[ToolTier, TierDefinition] = {
    ToolTier.T0: TierDefinition(
        tier=ToolTier.T0,
        label="Personal",
        description="Any tool used by a single person.",
        support_level="None — user is responsible.",
        monitoring="Scorecard data collected silently.",
    ),
    ToolTier.T1: TierDefinition(
        tier=ToolTier.T1,
        label="Shared (Registered)",
        description="Tool used by 2+ people or flagged for growing complexity.",
        support_level="Listed in catalog, basic docs expected.",
        monitoring="Scorecard data visible in dashboard.",
    ),
    ToolTier.T2: TierDefinition(
        tier=ToolTier.T2,
        label="Team (Owned)",
        description="Tool used by a team or tied to a business process.",
        support_level="Assigned tech owner, code reviewed, tests required.",
        monitoring="Quality trends tracked, alerts on regression.",
    ),
    ToolTier.T3: TierDefinition(
        tier=ToolTier.T3,
        label="Critical (Production)",
        description="Tool that business operations depend on.",
        support_level="Full production standards, CI/CD, monitoring, SLA.",
        monitoring="Full observability, alerting, on-call rotation.",
    ),
}


# Ordered transitions (each tuple is from_tier, to_tier)
VALID_TRANSITIONS: list[tuple[ToolTier, ToolTier]] = [
    (ToolTier.T0, ToolTier.T1),
    (ToolTier.T1, ToolTier.T2),
    (ToolTier.T2, ToolTier.T3),
]


def is_valid_transition(from_tier: ToolTier, to_tier: ToolTier) -> bool:
    """Check if a tier transition is valid (must be one step up)."""
    return (from_tier, to_tier) in VALID_TRANSITIONS


def get_tier_definition(tier: ToolTier) -> TierDefinition:
    """Get the definition for a tier."""
    return TIER_DEFINITIONS[tier]


def tier_index(tier: ToolTier) -> int:
    """Get the numeric index of a tier (T0=0, T1=1, etc.)."""
    return list(ToolTier).index(tier)
