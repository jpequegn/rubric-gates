"""Training data pipeline from scorecard data.

Converts stored ScoreResults into training-ready datasets and provides
synthetic data generation for bootstrapping when real data is limited.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from shared.models import Dimension
from shared.storage import StorageBackend


@dataclass
class TrainingExample:
    """A single training example: code + scores."""

    code: str
    composite_score: float
    dimension_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ScorecardDataset:
    """Load and transform scorecard data for training.

    Converts ScoreResult objects into TrainingExample format suitable
    for model training (distillation, GRPO, etc.).
    """

    def __init__(self) -> None:
        self._examples: list[TrainingExample] = []

    @property
    def examples(self) -> list[TrainingExample]:
        return list(self._examples)

    def __len__(self) -> int:
        return len(self._examples)

    def from_storage(
        self,
        storage: StorageBackend,
        min_scores: int = 0,
    ) -> list[TrainingExample]:
        """Convert stored ScoreResults into training examples.

        Args:
            storage: Storage backend to query.
            min_scores: Minimum number of scores required.

        Returns:
            List of training examples.

        Raises:
            ValueError: If fewer than min_scores results are found.
        """
        results = storage.query()

        if len(results) < min_scores:
            raise ValueError(
                f"Insufficient data: {len(results)} scores found, minimum {min_scores} required."
            )

        examples: list[TrainingExample] = []
        for result in results:
            dim_scores = {ds.dimension.value: ds.score for ds in result.dimension_scores}
            example = TrainingExample(
                code="",  # Code not stored in ScoreResult
                composite_score=result.composite_score,
                dimension_scores=dim_scores,
                metadata={
                    "user": result.user,
                    "timestamp": result.timestamp.isoformat(),
                    "skill_used": result.skill_used,
                },
            )
            examples.append(example)

        self._examples = examples
        return examples

    def generate_synthetic(
        self,
        num_examples: int = 1000,
        seed: int = 42,
    ) -> list[TrainingExample]:
        """Generate synthetic training data for bootstrapping.

        Produces realistic code + score pairs across the quality spectrum.

        Args:
            num_examples: Number of examples to generate.
            seed: Random seed for reproducibility.

        Returns:
            List of synthetic training examples.
        """
        rng = random.Random(seed)
        examples: list[TrainingExample] = []

        templates = _CODE_TEMPLATES

        for i in range(num_examples):
            # Pick a quality tier
            tier = rng.choices(
                ["high", "medium", "low"],
                weights=[0.3, 0.5, 0.2],
                k=1,
            )[0]

            template = rng.choice(templates[tier])
            dim_scores = _generate_dim_scores(rng, tier)
            composite = sum(dim_scores.values()) / len(dim_scores)

            example = TrainingExample(
                code=template,
                composite_score=round(composite, 3),
                dimension_scores={k: round(v, 3) for k, v in dim_scores.items()},
                metadata={"synthetic": True, "tier": tier, "index": i},
            )
            examples.append(example)

        self._examples = examples
        return examples

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> tuple[list[TrainingExample], list[TrainingExample]]:
        """Split examples into train/eval sets.

        Args:
            train_ratio: Fraction for training (rest is eval).
            seed: Random seed.

        Returns:
            Tuple of (train_examples, eval_examples).
        """
        rng = random.Random(seed)
        shuffled = list(self._examples)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        return shuffled[:split_idx], shuffled[split_idx:]

    def to_dicts(self) -> list[dict[str, Any]]:
        """Convert examples to list of dicts (for HuggingFace datasets)."""
        return [
            {
                "code": e.code,
                "composite_score": e.composite_score,
                **{f"score_{k}": v for k, v in e.dimension_scores.items()},
            }
            for e in self._examples
        ]


def _generate_dim_scores(rng: random.Random, tier: str) -> dict[str, float]:
    """Generate dimension scores for a given quality tier."""
    ranges = {
        "high": (0.7, 1.0),
        "medium": (0.4, 0.8),
        "low": (0.1, 0.5),
    }
    lo, hi = ranges[tier]
    return {dim.value: max(0.0, min(1.0, rng.uniform(lo, hi))) for dim in Dimension}


# --- Synthetic Code Templates ---

_CODE_TEMPLATES: dict[str, list[str]] = {
    "high": [
        'def calculate_total(items: list[dict]) -> float:\n    """Calculate total price with tax."""\n    return sum(item["price"] * (1 + item.get("tax_rate", 0.0)) for item in items)\n',
        'def validate_email(email: str) -> bool:\n    """Validate email format."""\n    import re\n    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$"\n    return bool(re.match(pattern, email))\n',
        'class DataProcessor:\n    """Processes incoming data with validation."""\n\n    def __init__(self, config: dict) -> None:\n        self.config = config\n\n    def process(self, data: list[dict]) -> list[dict]:\n        return [self._transform(item) for item in data if self._validate(item)]\n\n    def _validate(self, item: dict) -> bool:\n        return all(k in item for k in self.config.get("required_fields", []))\n\n    def _transform(self, item: dict) -> dict:\n        return {k: v.strip() if isinstance(v, str) else v for k, v in item.items()}\n',
    ],
    "medium": [
        "def process_data(data):\n    result = []\n    for item in data:\n        if item:\n            result.append(item)\n    return result\n",
        "def get_user(id):\n    users = load_users()\n    for u in users:\n        if u['id'] == id:\n            return u\n    return None\n",
        "class Handler:\n    def handle(self, request):\n        try:\n            result = self.do_work(request)\n            return result\n        except Exception:\n            return None\n\n    def do_work(self, request):\n        return request\n",
    ],
    "low": [
        "x = 1\ny = 2\nz = x + y\nprint(z)\n",
        'password = "admin123"\ndef login(user, pw):\n    if pw == password:\n        return True\n',
        'def query(name):\n    q = f"SELECT * FROM users WHERE name = {name}"\n    return q\n',
    ],
}
