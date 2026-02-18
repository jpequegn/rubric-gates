# rubric-gates

[![CI](https://github.com/jpequegn/rubric-gates/actions/workflows/ci.yml/badge.svg)](https://github.com/jpequegn/rubric-gates/actions/workflows/ci.yml)

Rubric-based quality gates for AI-generated code. Built for IT teams managing Claude Code skill ecosystems that need quality control without blocking productivity.

## Projects

### Project 1: Code Quality Scorecard (`scorecard/`)

Non-blocking quality scoring for every AI code generation. Scores across 5 dimensions (correctness, security, maintainability, documentation, testability) and surfaces trends via dashboard.

### Project 2: Security & Criticality Gate (`gate/`)

Tiered intervention system (green/yellow/red) that catches high-risk patterns — hardcoded credentials, injection vectors, unsafe operations — while letting safe code flow through.

### Project 3: Tool Registry & Graduation Path (`registry/`)

Lifecycle management for AI-generated tools. Automatic detection of when personal tools become shared/critical, with rubric-driven graduation from T0 (personal) to T3 (production critical).

### Project 4: Rubric-Trained Models (`training/`)

Closes the RLVR loop — uses rubric scores as reward signals to train models. Three phases:
- **4A: Scorer Distillation** — Train a fast local model to predict rubric scores, replacing expensive LLM-judge API calls
- **4B: Code Generation** — Fine-tune a code model with GRPO so it generates code that inherently passes your quality gates
- **4C: Skill Prompt Optimization** — Use rubric rewards to automatically optimize Claude Code skill prompts

## Architecture

```
[User invokes Claude Code skill]
    → [Skill generates code]
    → [Scorecard scores output (P1)]
    → [Gate evaluates tier (P2)]
    → [Registry tracks lifecycle (P3)]
    → [Dashboard surfaces insights]
    → [Scores feed back as training rewards (P4)]
```

## Design Principles

- **Non-blocking by default** — don't slow productive users
- **Visibility before gates** — see what's happening before restricting
- **Progressive enforcement** — warn first, gate only critical issues
- **Extensible** — designed for adding new quality and security gates

## Getting Started

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run scorecard on a file
uv run python -m scorecard.score path/to/file.py
```

## Structure

```
rubric-gates/
├── scorecard/          # P1: Quality scoring engine
│   ├── dimensions/     # Individual dimension scorers
│   ├── hooks/          # Claude Code hook integration
│   └── dashboard/      # Monitoring dashboard
├── gate/               # P2: Security & criticality gate
│   ├── patterns/       # Critical pattern detectors
│   ├── tiers/          # Green/yellow/red tier logic
│   └── overrides/      # Override workflow & audit
├── registry/           # P3: Tool registry & graduation
│   ├── catalog/        # Tool catalog storage
│   ├── graduation/     # Graduation rubrics & triggers
│   └── workflows/      # Graduation workflow engine
├── training/           # P4: Rubric-trained models
│   ├── distillation.py # Phase A: Scorer distillation
│   ├── code_gen.py     # Phase B: GRPO code generation
│   └── skill_optimizer.py # Phase C: Skill prompt optimization
├── shared/             # Shared utilities
│   ├── config.py       # Configuration management
│   └── models.py       # Common data models
├── tests/              # Test suites
└── docs/               # Documentation
    └── plans/          # Implementation plans
```

## License

MIT
