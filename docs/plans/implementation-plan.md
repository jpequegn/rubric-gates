# Rubric Engineering Projects for IT-Managed AI Tooling

**Context:** Medium-sized Claude Code skill ecosystem (10-30 skills) used by technical-adjacent business users (data analysts, PMs) to build personal productivity tools — replacing what historically would have been Excel spreadsheets. Minimal QC today. Tech team discovers issues after the fact.

**Core Problem:** AI-generated personal tools grow in importance without visibility, quality control, or ownership — creating shadow IT risk with code instead of spreadsheets.

**Design Principles:**
- Non-blocking by default — don't slow productive users down
- Visibility before gates — see what's happening before you restrict it
- Progressive enforcement — warn first, gate only what's critical
- Respect the user — these are technical-adjacent people, not adversaries

---

## Project 1: Code Quality Scorecard

**Goal:** Give the tech team visibility into what's being generated, how often, and at what quality level — without blocking anyone.

**Rubric Techniques Used:**
- Multi-dimensional rubrics (from Projects 1, 7, 17)
- LLM-as-judge for subjective dimensions (from Project 3)

### What It Does

Every time a Claude Code skill generates or modifies code, a lightweight rubric scores the output across 5 dimensions:

| Dimension | Weight | What It Measures | Method |
|-----------|--------|-----------------|--------|
| **Correctness** | 25% | Syntax valid, no obvious logic errors | AST parsing + static analysis |
| **Security** | 25% | No hardcoded secrets, no injection vectors, safe dependencies | Rule-based checks (Bandit/Semgrep-style) |
| **Maintainability** | 20% | Reasonable complexity, modular structure, naming quality | Cyclomatic complexity + LLM judge |
| **Documentation** | 15% | Comments on non-obvious logic, README if multi-file | LLM judge |
| **Testability** | 15% | Could this be tested? Are there test cases? | LLM judge |

Scores are logged (not displayed to the user by default) and aggregated into a dashboard the tech team can monitor.

### Architecture

```
[User invokes skill]
    → [Skill generates code]
    → [Post-generation hook runs rubric scorer]
    → [Scores logged to central store (JSON/SQLite)]
    → [Dashboard reads from store]
```

### Key Deliverables

1. **Rubric scorer module** — Python module that takes generated code + context, returns scores per dimension
2. **Claude Code hook integration** — Post-edit hook that triggers scoring (non-blocking, runs async)
3. **Score storage** — Append-only log (JSONL or SQLite) with: timestamp, user, skill used, file(s) touched, scores per dimension, composite score
4. **Dashboard** — Simple read-only view showing:
   - Score trends over time (are things getting better or worse?)
   - Distribution by user (who generates the most code? at what quality?)
   - Distribution by skill (which skills produce the best/worst code?)
   - Red flags (any scores below threshold)

### What This Unlocks

- Tech team can see the full landscape of AI-generated code for the first time
- Data to decide WHERE to invest in gates (Project 2)
- Baseline metrics to measure improvement
- Early warning when a personal tool is growing in complexity

### Estimated Scope

- Rubric scorer: 2-3 days
- Hook integration: 1 day
- Storage + dashboard: 2-3 days
- **Total: ~1 week**

---

## Project 2: Security & Criticality Gate

**Goal:** Automatically catch high-risk issues in generated code and route them for review — blocking only when necessary.

**Rubric Techniques Used:**
- Self-critique middleware (from Project 16)
- Reward hacking defense (from Projects 12, 22)
- Code execution verification (from Project 2)

**Prerequisite:** Project 1 (uses scorecard data to calibrate thresholds)

### What It Does

A tiered response system based on rubric scores:

| Tier | Trigger | Action |
|------|---------|--------|
| **Green** | All dimensions above threshold | No intervention. Score logged silently. |
| **Yellow** | Maintainability or documentation below threshold | User sees a brief advisory: "Consider adding error handling" or "This is getting complex — consider splitting into modules." Non-blocking. |
| **Red** | Security score below threshold OR critical pattern detected | Generation paused. User sees specific issue. Option to fix, override with justification, or escalate to tech team. |

### Critical Pattern Detection (Red Tier)

These always trigger regardless of composite score:

- **Hardcoded credentials** — API keys, passwords, connection strings in code
- **SQL injection vectors** — Unsanitized user input in queries
- **Unsafe file operations** — Unrestricted path access, shell injection
- **Dependency risk** — Installing unvetted packages, pinning to `*` versions
- **Data exposure** — Logging PII, writing sensitive data to unprotected files
- **Scope creep** — Single file exceeding complexity threshold (the "2000-line script" problem)

### Anti-Gaming Measures

Drawing from Project 12's red-team findings, the rubric defends against:

| Gaming Pattern | Defense |
|----------------|---------|
| Adding empty comments to inflate documentation score | Semantic density check — comments must be meaningful |
| Wrapping everything in try/catch for error handling score | Check that caught exceptions are actually handled, not swallowed |
| Splitting into many trivial files to lower per-file complexity | Measure total complexity across the tool, not per-file |
| Adding unused test stubs | Verify tests actually assert something meaningful |

### Architecture

```
[User invokes skill]
    → [Skill generates code]
    → [Pre-commit hook runs gate checker]
    → [Green?] → Log + continue
    → [Yellow?] → Show advisory + continue
    → [Red?] → Block + show issue + offer remediation paths
```

### Key Deliverables

1. **Gate checker module** — Extends Project 1's scorer with tier logic and critical pattern detection
2. **Advisory message templates** — Friendly, specific, actionable messages for yellow-tier issues
3. **Red-tier workflow** — Block UI, show issue, offer: auto-fix / manual fix / override-with-justification / escalate
4. **Override audit log** — When users override red-tier blocks, log the justification for tech team review
5. **Threshold configuration** — YAML config per skill or per team, so tech team can tune sensitivity

### Configuration Example

```yaml
# .rubric-gate.yaml
thresholds:
  green:
    min_composite: 0.7
  yellow:
    min_composite: 0.5
    dimensions:
      maintainability: 0.4
      documentation: 0.3
  red:
    security: 0.3  # Below this = always block

critical_patterns:
  - hardcoded_credentials
  - sql_injection
  - unsafe_file_ops
  - unvetted_dependencies

overrides:
  require_justification: true
  notify_tech_team: true
  max_overrides_per_user_per_week: 3
```

### Estimated Scope

- Gate checker + tier logic: 3-4 days
- Critical pattern detectors: 3-4 days
- Advisory/block UI integration: 2 days
- Override workflow + audit: 2 days
- Configuration system: 1 day
- **Total: ~2-3 weeks**

---

## Project 3: Tool Registry & Graduation Path

**Goal:** Create a structured path for personal tools to "graduate" from ad-hoc to team-supported, using rubrics to determine readiness.

**Rubric Techniques Used:**
- Skill/prompt validation (from Project 18)
- Human-in-the-loop refinement (from Project 24)
- Benchmark suite patterns (from Project 20)
- A/B testing for rubric tuning (from Project 30)

**Prerequisite:** Projects 1 and 2 (uses scorecard history and gate data)

### The Problem This Solves

The lifecycle of a business-user tool today:

```
1. User builds personal tool with Claude Code
2. Tool works well, user relies on it daily
3. Colleagues notice, start using it too
4. Tool becomes business-critical
5. Original user changes roles / leaves
6. Tool breaks, nobody understands it
7. Tech team scrambles to reverse-engineer and fix
```

This project inserts a structured "graduation" process between steps 3 and 4.

### Graduation Tiers

| Tier | Name | Criteria | Support Level |
|------|------|----------|---------------|
| **T0** | Personal | Any tool with < 1 user | None — user's responsibility |
| **T1** | Shared | 2+ users OR scorecard flags growing complexity | Registered in catalog. Basic documentation required. |
| **T2** | Team | Used by a team OR tied to a business process | Code reviewed by tech team. Tests required. Assigned tech owner. |
| **T3** | Critical | Business-critical OR handles sensitive data | Full production standards. CI/CD. Monitoring. SLA. |

### Graduation Rubric

When a tool is nominated for graduation (automatically or manually), a comprehensive rubric evaluates readiness:

**T0 → T1 (Registration)**
- [ ] Tool has a description of what it does (README or header comment)
- [ ] No red-tier security issues in scorecard history
- [ ] User confirms they want to share it

**T1 → T2 (Team Adoption)**
- [ ] Composite quality score > 0.7 (from Project 1 scorecard)
- [ ] Zero unresolved red-tier issues (from Project 2 gate)
- [ ] Has at least basic tests (even generated ones)
- [ ] Code reviewed by at least one tech team member
- [ ] Tech owner assigned
- [ ] Dependencies are vetted and pinned

**T2 → T3 (Production Critical)**
- [ ] Full test coverage (unit + integration)
- [ ] Security review completed
- [ ] Error handling and logging in place
- [ ] Rollback plan documented
- [ ] Monitoring/alerting configured
- [ ] Knowledge transfer session held with tech team

### Automatic Graduation Triggers

The system monitors scorecard data and suggests graduation when:

- **T0 → T1:** Tool is accessed by a second user, OR total lines of code exceed 500, OR scorecard shows increasing complexity trend
- **T1 → T2:** Tool is used daily for 2+ weeks by 3+ users, OR tool touches sensitive data, OR tool integrates with production systems
- **T2 → T3:** Tool failure would impact business operations (detected via usage patterns and user-reported criticality)

### Tool Registry

A catalog of all registered tools (T1+):

```yaml
# registry/tools/expense-categorizer.yaml
name: Expense Categorizer
description: Categorizes expense reports by department and GL code
tier: T1
created_by: jane.doe
created_date: 2026-01-15
tech_owner: null  # Assigned at T2
users:
  - jane.doe
  - bob.smith
repository: null  # Link added at T2
scorecard:
  latest_composite: 0.72
  trend: improving
  red_flags: 0
graduation_history:
  - from: T0
    to: T1
    date: 2026-02-01
    reason: "Second user (bob.smith) started using tool"
```

### Key Deliverables

1. **Graduation rubric engine** — Evaluates tools against tier-specific criteria
2. **Automatic trigger system** — Monitors scorecard data for graduation signals
3. **Tool registry** — YAML/database catalog of all T1+ tools
4. **Graduation workflow** — Notification → rubric evaluation → action items → review → promotion
5. **Dashboard extension** — Add registry view to Project 1's dashboard showing tool inventory by tier
6. **Knowledge transfer template** — Structured format for documenting a tool when it reaches T2/T3

### Estimated Scope

- Graduation rubric engine: 3-4 days
- Automatic trigger system: 2-3 days
- Tool registry + CRUD: 3-4 days
- Graduation workflow: 3-4 days
- Dashboard extension: 2 days
- Templates + docs: 1-2 days
- **Total: ~3-4 weeks**

---

## Implementation Roadmap

```
Month 1:         [====== Project 1: Scorecard ======]
                                   |
Month 2-3:                         [======== Project 2: Gate ========]
                                                    |
Month 3-4:                                          [========= Project 3: Registry =========]
```

### Quick Wins Along the Way

- **Week 1:** Even before the dashboard, just logging scores to a JSONL file gives immediate visibility
- **Week 3:** Critical pattern detection (hardcoded creds, SQL injection) can ship independently as a simple hook
- **Week 6:** The graduation trigger for T0→T1 ("a second person is using this") is simple and immediately valuable

### Success Metrics

| Metric | Baseline (today) | Target (6 months) |
|--------|-------------------|--------------------|
| Visibility into AI-generated tools | 0% | 100% of skill-generated code scored |
| Security issues caught pre-deployment | ~0 | 90%+ of critical patterns caught |
| Tools with assigned tech owner | Unknown | 100% of T2+ tools |
| Mean time to discover critical tool | Weeks/months (reactive) | < 24 hours (proactive) |
| Business user satisfaction | N/A | No decrease (non-blocking design) |
