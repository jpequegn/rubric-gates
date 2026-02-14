"""Knowledge transfer document generator.

Produces a Markdown document from a tool's registry entry, source code,
and scorecard history. Designed for tech team onboarding when a tool
graduates to T2 or T3.

All analysis is rule-based (no LLM dependency). Sections are generated
from code inspection and scorecard data.
"""

from __future__ import annotations

import ast
from pathlib import Path

from shared.models import ScoreResult, ToolRegistryEntry
from shared.storage import StorageBackend


@staticmethod
def _safe_parse(code: str) -> ast.Module | None:
    """Parse code, returning None on syntax error."""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


class CodeAnalysis:
    """Static analysis results for a source file."""

    def __init__(self, path: Path, code: str) -> None:
        self.path = path
        self.code = code
        self.lines = code.splitlines()
        self.line_count = len(self.lines)
        self.tree = _safe_parse(code)

        self.functions: list[str] = []
        self.classes: list[str] = []
        self.imports: list[str] = []
        self.stdlib_imports: list[str] = []
        self.third_party_imports: list[str] = []

        if self.tree:
            self._analyze()

    def _analyze(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.FunctionDef):
                self.functions.append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                self.functions.append(f"{node.name} (async)")
            elif isinstance(node, ast.ClassDef):
                self.classes.append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.imports.append(node.module)

        # Classify imports (heuristic: dotted names with common prefixes = stdlib)
        _STDLIB_PREFIXES = {
            "os",
            "sys",
            "re",
            "json",
            "csv",
            "math",
            "datetime",
            "pathlib",
            "collections",
            "itertools",
            "functools",
            "typing",
            "dataclasses",
            "enum",
            "abc",
            "io",
            "logging",
            "argparse",
            "unittest",
            "ast",
            "hashlib",
            "uuid",
            "time",
            "shutil",
            "subprocess",
            "tempfile",
        }
        for imp in self.imports:
            root = imp.split(".")[0]
            if root in _STDLIB_PREFIXES:
                self.stdlib_imports.append(imp)
            else:
                self.third_party_imports.append(imp)


class KnowledgeTransferGenerator:
    """Generates knowledge transfer Markdown documents.

    Args:
        storage: Score storage backend for history queries.
    """

    def __init__(self, storage: StorageBackend | None = None) -> None:
        self._storage = storage

    def generate(self, tool: ToolRegistryEntry) -> str:
        """Generate a knowledge transfer document as Markdown.

        Args:
            tool: The tool registry entry.

        Returns:
            Markdown string.
        """
        sections = [
            self._header(tool),
            self._overview(tool),
            self._architecture(tool),
            self._quality_profile(tool),
            self._risk_assessment(tool),
            self._maintenance_guide(tool),
            self._scorecard_history(tool),
        ]
        return "\n\n".join(s for s in sections if s) + "\n"

    def _header(self, tool: ToolRegistryEntry) -> str:
        return f"# Knowledge Transfer: {tool.name}"

    def _overview(self, tool: ToolRegistryEntry) -> str:
        lines = ["## Overview", ""]
        lines.append(f"- **Purpose:** {tool.description or 'Not documented'}")
        lines.append(f"- **Created by:** {tool.created_by or 'Unknown'}")
        lines.append(f"- **Current tier:** {tool.tier.value}")
        lines.append(f"- **Tech owner:** {tool.tech_owner or 'Unassigned'}")

        if tool.users:
            lines.append(f"- **Users:** {', '.join(tool.users)}")
        if tool.tags:
            lines.append(f"- **Tags:** {', '.join(tool.tags)}")

        return "\n".join(lines)

    def _architecture(self, tool: ToolRegistryEntry) -> str:
        lines = ["## Architecture", ""]

        analysis = self._analyze_source(tool)
        if analysis is None:
            lines.append("*Source code not available for analysis.*")
            return "\n".join(lines)

        lines.append(f"- **Source:** `{tool.source_path}`")
        lines.append(f"- **Lines of code:** {analysis.line_count}")

        if analysis.classes:
            lines.append(f"- **Classes:** {', '.join(analysis.classes)}")
        if analysis.functions:
            func_display = analysis.functions[:10]
            suffix = (
                f" (+{len(analysis.functions) - 10} more)" if len(analysis.functions) > 10 else ""
            )
            lines.append(f"- **Functions:** {', '.join(func_display)}{suffix}")

        if analysis.third_party_imports:
            lines.append("")
            lines.append("### Dependencies")
            lines.append("")
            for imp in sorted(set(analysis.third_party_imports)):
                lines.append(f"- `{imp}`")

        return "\n".join(lines)

    def _quality_profile(self, tool: ToolRegistryEntry) -> str:
        lines = ["## Quality Profile", ""]
        sc = tool.scorecard

        lines.append(f"- **Latest composite score:** {sc.latest_composite:.2f}")
        lines.append(f"- **Trend:** {sc.trend}")
        lines.append(f"- **Total scoring events:** {sc.total_scores}")

        if sc.latest_scores:
            sorted_dims = sorted(sc.latest_scores.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_dims) >= 2:
                top = sorted_dims[:2]
                bottom = sorted_dims[-2:]
                lines.append(
                    f"- **Strongest dimensions:** "
                    f"{top[0][0]} ({top[0][1]:.2f}), {top[1][0]} ({top[1][1]:.2f})"
                )
                lines.append(
                    f"- **Areas for improvement:** "
                    f"{bottom[0][0]} ({bottom[0][1]:.2f}), {bottom[1][0]} ({bottom[1][1]:.2f})"
                )

        if sc.red_flags > 0:
            lines.append(f"- **Security findings:** {sc.red_flags} red-tier issue(s)")
        else:
            lines.append("- **Security findings:** None")

        return "\n".join(lines)

    def _risk_assessment(self, tool: ToolRegistryEntry) -> str:
        lines = ["## Risk Assessment", ""]
        sc = tool.scorecard

        # Test coverage
        test_score = sc.latest_scores.get("testability", 0.0)
        if test_score >= 0.8:
            lines.append("- **Test coverage:** Comprehensive")
        elif test_score >= 0.5:
            lines.append("- **Test coverage:** Basic")
        else:
            lines.append("- **Test coverage:** Insufficient or absent")

        # Bus factor
        bus_factor = len(tool.users)
        if bus_factor <= 1:
            lines.append("- **Bus factor:** 1 (high risk — single point of knowledge)")
        else:
            lines.append(f"- **Bus factor:** {bus_factor}")

        # Complexity from source
        analysis = self._analyze_source(tool)
        if analysis:
            if analysis.line_count > 500:
                lines.append(f"- **Complexity:** High ({analysis.line_count} lines)")
            elif analysis.line_count > 200:
                lines.append(f"- **Complexity:** Moderate ({analysis.line_count} lines)")
            else:
                lines.append(f"- **Complexity:** Low ({analysis.line_count} lines)")

            # Single point of failure
            if len(analysis.classes) <= 1 and len(analysis.functions) > 10:
                lines.append("- **Single point of failure:** Yes — many functions in one file")

        return "\n".join(lines)

    def _maintenance_guide(self, tool: ToolRegistryEntry) -> str:
        lines = ["## Maintenance Guide", ""]

        if tool.source_path:
            lines.append(f"- **Source location:** `{tool.source_path}`")

        # How to run
        analysis = self._analyze_source(tool)
        if analysis:
            has_main = any("__main__" in line for line in analysis.lines)
            if has_main:
                lines.append(f"- **How to run:** `python {tool.source_path}`")
            has_cli = any("argparse" in imp for imp in analysis.imports)
            if has_cli:
                lines.append(f"- **CLI:** `python {tool.source_path} --help`")

        # Known issues from scorecard
        sc = tool.scorecard
        weak_dims = [(dim, score) for dim, score in sc.latest_scores.items() if score < 0.5]
        if weak_dims:
            lines.append("")
            lines.append("### Known Issues")
            lines.append("")
            for dim, score in weak_dims:
                lines.append(f"- **{dim}** score is low ({score:.2f}) — needs improvement")

        return "\n".join(lines)

    def _scorecard_history(self, tool: ToolRegistryEntry) -> str:
        lines = ["## Scorecard History", ""]

        if not self._storage:
            lines.append("*Score history not available (no storage backend).*")
            return "\n".join(lines)

        scores = self._get_tool_scores(tool)
        if not scores:
            lines.append("*No scoring events recorded.*")
            return "\n".join(lines)

        # Summary table
        lines.append("| Date | Score | User |")
        lines.append("|------|-------|------|")
        for s in sorted(scores, key=lambda x: x.timestamp, reverse=True)[:20]:
            date_str = s.timestamp.strftime("%Y-%m-%d")
            lines.append(f"| {date_str} | {s.composite_score:.2f} | {s.user} |")

        if len(scores) > 20:
            lines.append(f"| ... | *{len(scores) - 20} more* | |")

        return "\n".join(lines)

    def _analyze_source(self, tool: ToolRegistryEntry) -> CodeAnalysis | None:
        """Analyze the tool's source code if available."""
        if not tool.source_path:
            return None
        path = Path(tool.source_path)
        if not path.exists() or not path.is_file():
            return None
        try:
            code = path.read_text()
        except Exception:
            return None
        return CodeAnalysis(path, code)

    def _get_tool_scores(self, tool: ToolRegistryEntry) -> list[ScoreResult]:
        """Get scoring history for a tool."""
        if not self._storage:
            return []
        all_scores = self._storage.query()
        if not tool.source_path:
            return []
        return [s for s in all_scores if any(tool.source_path in f for f in s.files_touched)]
