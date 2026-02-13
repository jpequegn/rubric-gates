"""Tests for security dimension scorer."""

import ast

import pytest

from scorecard.dimensions.security import (
    SecurityScorer,
    detect_eval_exec,
    detect_hardcoded_credentials,
    detect_insecure_http,
    detect_sensitive_paths,
    detect_shell_injection,
    detect_sql_injection,
    detect_unsafe_deserialization,
)
from shared.models import Dimension, ScoringMethod


@pytest.fixture()
def scorer():
    return SecurityScorer()


def _parse(code: str) -> ast.Module:
    return ast.parse(code)


# --- Hardcoded Credentials ---


class TestHardcodedCredentials:
    def test_no_credentials(self):
        code = "x = 1\nname = 'hello'"
        assert detect_hardcoded_credentials(code) == []

    def test_api_key_assignment(self):
        code = 'api_key = "AKIAIOSFODNN7EXAMPLE1"'
        findings = detect_hardcoded_credentials(code)
        assert len(findings) == 1
        assert findings[0].pattern == "hardcoded_credential"
        assert findings[0].severity.value == "critical"

    def test_password_assignment(self):
        code = "password = 'super_secret_password_123'"
        findings = detect_hardcoded_credentials(code)
        assert len(findings) == 1

    def test_secret_key_assignment(self):
        code = 'secret_key = "abcdef1234567890abcdef"'
        findings = detect_hardcoded_credentials(code)
        assert len(findings) == 1

    def test_aws_access_key(self):
        code = 'key = "AKIAIOSFODNN7EXAMPLE"'
        findings = detect_hardcoded_credentials(code)
        assert len(findings) >= 1

    def test_env_var_lookup_ok(self):
        code = 'api_key = os.environ["API_KEY"]'
        assert detect_hardcoded_credentials(code) == []

    def test_comment_ignored(self):
        code = '# api_key = "sk-1234567890abcdef"'
        assert detect_hardcoded_credentials(code) == []

    def test_short_values_ignored(self):
        code = 'password = "short"'
        assert detect_hardcoded_credentials(code) == []

    def test_token_hex_string(self):
        code = 'token = "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2"'
        findings = detect_hardcoded_credentials(code)
        assert len(findings) >= 1


# --- Sensitive Paths ---


class TestSensitivePaths:
    def test_no_sensitive_paths(self):
        code = 'path = "/usr/local/bin/python"'
        assert detect_sensitive_paths(code) == []

    def test_etc_passwd(self):
        code = 'f = open("/etc/passwd")'
        findings = detect_sensitive_paths(code)
        assert len(findings) == 1
        assert findings[0].pattern == "sensitive_path"
        assert findings[0].severity.value == "high"

    def test_etc_shadow(self):
        code = 'open("/etc/shadow")'
        findings = detect_sensitive_paths(code)
        assert len(findings) == 1

    def test_ssh_directory(self):
        code = 'key = open("~/.ssh/id_rsa").read()'
        findings = detect_sensitive_paths(code)
        assert len(findings) == 1

    def test_aws_credentials(self):
        code = 'config = open("~/.aws/credentials")'
        findings = detect_sensitive_paths(code)
        assert len(findings) == 1

    def test_dotenv(self):
        code = 'load("~/.env")'
        findings = detect_sensitive_paths(code)
        assert len(findings) == 1

    def test_comment_ignored(self):
        code = '# open("/etc/passwd")'
        assert detect_sensitive_paths(code) == []


# --- Insecure HTTP ---


class TestInsecureHTTP:
    def test_no_urls(self):
        code = "x = 1"
        assert detect_insecure_http(code) == []

    def test_https_ok(self):
        code = 'url = "https://example.com/api"'
        assert detect_insecure_http(code) == []

    def test_http_external(self):
        code = 'url = "http://example.com/api"'
        findings = detect_insecure_http(code)
        assert len(findings) == 1
        assert findings[0].pattern == "insecure_http"
        assert findings[0].severity.value == "medium"

    def test_localhost_ok(self):
        code = 'url = "http://localhost:8080/api"'
        assert detect_insecure_http(code) == []

    def test_127_ok(self):
        code = 'url = "http://127.0.0.1:3000"'
        assert detect_insecure_http(code) == []

    def test_comment_ignored(self):
        code = '# url = "http://example.com"'
        assert detect_insecure_http(code) == []


# --- SQL Injection ---


class TestSQLInjection:
    def test_no_sql(self):
        tree = _parse('x = f"hello {name}"')
        assert detect_sql_injection(tree) == []

    def test_fstring_select(self):
        code = 'query = f"SELECT * FROM users WHERE id = {user_id}"'
        tree = _parse(code)
        findings = detect_sql_injection(tree)
        assert len(findings) == 1
        assert findings[0].pattern == "sql_injection"
        assert findings[0].severity.value == "critical"

    def test_fstring_insert(self):
        code = 'q = f"INSERT INTO logs VALUES ({val})"'
        tree = _parse(code)
        findings = detect_sql_injection(tree)
        assert len(findings) == 1

    def test_format_select(self):
        code = '"SELECT * FROM users WHERE name = {}".format(name)'
        tree = _parse(code)
        findings = detect_sql_injection(tree)
        assert len(findings) == 1

    def test_percent_format(self):
        code = 'q = "DELETE FROM users WHERE id = %s" % user_id'
        tree = _parse(code)
        findings = detect_sql_injection(tree)
        assert len(findings) == 1

    def test_parameterized_query_ok(self):
        code = 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))'
        tree = _parse(code)
        assert detect_sql_injection(tree) == []

    def test_constant_sql_ok(self):
        code = 'query = "SELECT * FROM users WHERE active = 1"'
        tree = _parse(code)
        assert detect_sql_injection(tree) == []


# --- Shell Injection ---


class TestShellInjection:
    def test_no_shell(self):
        tree = _parse("x = 1")
        assert detect_shell_injection(tree) == []

    def test_os_system(self):
        code = 'os.system("rm -rf /")'
        tree = _parse(code)
        findings = detect_shell_injection(tree)
        assert len(findings) == 1
        assert findings[0].pattern == "shell_injection"
        assert findings[0].severity.value == "critical"
        assert "os.system()" in findings[0].description

    def test_subprocess_shell_true(self):
        code = "subprocess.run(cmd, shell=True)"
        tree = _parse(code)
        findings = detect_shell_injection(tree)
        assert len(findings) == 1
        assert "shell=True" in findings[0].description

    def test_subprocess_popen_shell_true(self):
        code = "subprocess.Popen(cmd, shell=True)"
        tree = _parse(code)
        findings = detect_shell_injection(tree)
        assert len(findings) == 1

    def test_subprocess_shell_false_ok(self):
        code = 'subprocess.run(["ls", "-la"], shell=False)'
        tree = _parse(code)
        assert detect_shell_injection(tree) == []

    def test_subprocess_no_shell_ok(self):
        code = 'subprocess.run(["ls", "-la"])'
        tree = _parse(code)
        assert detect_shell_injection(tree) == []

    def test_check_output_shell_true(self):
        code = 'subprocess.check_output("ls", shell=True)'
        tree = _parse(code)
        findings = detect_shell_injection(tree)
        assert len(findings) == 1


# --- eval/exec ---


class TestEvalExec:
    def test_no_eval(self):
        tree = _parse("x = 1")
        assert detect_eval_exec(tree) == []

    def test_eval(self):
        code = "result = eval(user_input)"
        tree = _parse(code)
        findings = detect_eval_exec(tree)
        assert len(findings) == 1
        assert findings[0].pattern == "eval_exec"
        assert findings[0].severity.value == "high"
        assert "eval()" in findings[0].description

    def test_exec(self):
        code = "exec(code_string)"
        tree = _parse(code)
        findings = detect_eval_exec(tree)
        assert len(findings) == 1
        assert "exec()" in findings[0].description

    def test_both_eval_and_exec(self):
        code = "eval(a)\nexec(b)"
        tree = _parse(code)
        findings = detect_eval_exec(tree)
        assert len(findings) == 2

    def test_eval_as_method_ok(self):
        code = "model.eval()"
        tree = _parse(code)
        # model.eval() is an attribute call, not a Name call
        assert detect_eval_exec(tree) == []


# --- Unsafe Deserialization ---


class TestUnsafeDeserialization:
    def test_no_imports(self):
        tree = _parse("import os")
        assert detect_unsafe_deserialization(tree) == []

    def test_pickle_import(self):
        tree = _parse("import pickle")
        findings = detect_unsafe_deserialization(tree)
        assert len(findings) == 1
        assert findings[0].pattern == "unsafe_deserialization"
        assert findings[0].severity.value == "high"

    def test_marshal_import(self):
        tree = _parse("import marshal")
        findings = detect_unsafe_deserialization(tree)
        assert len(findings) == 1

    def test_shelve_import(self):
        tree = _parse("import shelve")
        findings = detect_unsafe_deserialization(tree)
        assert len(findings) == 1

    def test_from_pickle_import(self):
        tree = _parse("from pickle import loads")
        findings = detect_unsafe_deserialization(tree)
        assert len(findings) == 1

    def test_safe_imports_ok(self):
        tree = _parse("import json\nimport os\nfrom pathlib import Path")
        assert detect_unsafe_deserialization(tree) == []


# --- SecurityScorer Integration ---


class TestSecurityScorer:
    def test_clean_code(self, scorer):
        code = """\
import os
from pathlib import Path

def process(filename: str) -> str:
    path = Path(filename)
    if not path.exists():
        return ""
    with open(path) as f:
        return f.read()
"""
        score = scorer.score(code, "main.py")
        assert score.dimension == Dimension.SECURITY
        assert score.method == ScoringMethod.RULE_BASED
        assert score.score == 1.0
        assert score.metadata["finding_count"] == 0

    def test_empty_code(self, scorer):
        score = scorer.score("", "empty.py")
        assert score.score == 0.0
        assert "Empty" in score.details

    def test_whitespace_only(self, scorer):
        score = scorer.score("   \n\n  ", "empty.py")
        assert score.score == 0.0

    def test_critical_finding_penalty(self, scorer):
        code = 'api_key = "AKIAIOSFODNN7EXAMPLE1"'
        score = scorer.score(code, "bad.py")
        assert score.score == pytest.approx(0.7)  # 1.0 - 0.3

    def test_high_finding_penalty(self, scorer):
        code = "result = eval(user_input)"
        score = scorer.score(code, "bad.py")
        assert score.score == pytest.approx(0.8)  # 1.0 - 0.2

    def test_medium_finding_penalty(self, scorer):
        code = 'url = "http://example.com/api"'
        score = scorer.score(code, "bad.py")
        assert score.score == pytest.approx(0.9)  # 1.0 - 0.1

    def test_multiple_findings(self, scorer):
        code = """\
api_key = "AKIAIOSFODNN7EXAMPLE1"
os.system("rm -rf /")
result = eval(user_input)
url = "http://example.com/data"
"""
        score = scorer.score(code, "bad.py")
        # 1.0 - 0.3 (credential) - 0.3 (os.system) - 0.2 (eval) - 0.1 (http) = 0.1
        assert score.score == pytest.approx(0.1)
        assert score.metadata["finding_count"] == 4

    def test_floor_at_zero(self, scorer):
        code = """\
api_key = "AKIAIOSFODNN7EXAMPLE1"
secret_key = "abcdef1234567890abcdef1234567890ab"
password = "super_secret_password_123"
os.system("rm -rf /")
"""
        score = scorer.score(code, "very_bad.py")
        assert score.score == 0.0

    def test_syntax_error_still_runs_regex(self, scorer):
        code = 'api_key = "AKIAIOSFODNN7EXAMPLE1"\ndef broken( return'
        score = scorer.score(code, "bad.py")
        assert score.score < 1.0
        assert "hardcoded_credential" in score.details

    def test_deterministic(self, scorer):
        code = 'eval(x)\nurl = "http://example.com"'
        s1 = scorer.score(code, "t.py")
        s2 = scorer.score(code, "t.py")
        assert s1.score == s2.score

    def test_details_contain_line_numbers(self, scorer):
        code = 'api_key = "AKIAIOSFODNN7EXAMPLE1"'
        score = scorer.score(code, "test.py")
        assert "line 1" in score.details

    def test_safe_patterns_no_false_positives(self, scorer):
        code = """\
import json
import os
from pathlib import Path

def fetch_data(url: str) -> dict:
    response = requests.get(url)
    return response.json()

def run_command(args: list[str]) -> str:
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout

config = json.loads(Path("config.json").read_text())
api_key = os.environ.get("API_KEY", "")
"""
        score = scorer.score(code, "safe.py")
        assert score.score == 1.0
