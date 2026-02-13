"""Tests for critical pattern detectors."""

from gate.patterns import (
    get_all_detectors,
    scan_all,
)
from gate.patterns.credentials import CredentialsDetector
from gate.patterns.data_exposure import DataExposureDetector
from gate.patterns.dependencies import DependenciesDetector
from gate.patterns.file_ops import FileOpsDetector
from gate.patterns.shell_injection import ShellInjectionDetector
from gate.patterns.sql_injection import SQLInjectionDetector


# --- Registry ---


class TestRegistry:
    def test_all_detectors_registered(self):
        detectors = get_all_detectors()
        names = {d.name for d in detectors}
        assert "hardcoded_credentials" in names
        assert "sql_injection" in names
        assert "shell_injection" in names
        assert "unsafe_file_ops" in names
        assert "unvetted_dependencies" in names
        assert "data_exposure" in names

    def test_at_least_six_detectors(self):
        assert len(get_all_detectors()) >= 6

    def test_scan_all_runs_all(self):
        # Clean code should produce no findings
        findings = scan_all("x = 1\n", "clean.py")
        assert isinstance(findings, list)


# --- Credentials ---


class TestCredentialsDetector:
    def setup_method(self):
        self.detector = CredentialsDetector()

    def test_openai_key(self):
        code = 'api_key = "sk-abc123def456ghi789jklmnopqrstuvwxyz"\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) >= 1
        assert any("OpenAI" in f.description for f in findings)

    def test_aws_key(self):
        code = 'key = "AKIAIOSFODNN7EXAMPLE"\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) >= 1
        assert any("AWS" in f.description for f in findings)

    def test_github_token(self):
        code = 'token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklm"\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) >= 1
        assert any("GitHub" in f.description for f in findings)

    def test_slack_token(self):
        code = 'token = "xoxb-1234-5678-abcdefghijklmnopqrst"\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) >= 1

    def test_password_assignment(self):
        code = 'password = "supersecret123"\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) >= 1
        assert any(
            "password" in f.description.lower() or "secret" in f.description.lower()
            for f in findings
        )

    def test_api_key_assignment(self):
        code = 'api_key = "my-secret-key-12345"\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) >= 1

    def test_connection_string(self):
        code = 'db_url = "postgres://admin:password123@localhost/mydb"\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) >= 1
        assert any("connection string" in f.description.lower() for f in findings)

    def test_bearer_token(self):
        code = 'headers = {"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc123"}\n'
        findings = self.detector.detect(code, "api.py")
        assert len(findings) >= 1

    def test_env_var_ok(self):
        code = 'password = os.environ.get("DB_PASSWORD")\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) == 0

    def test_getenv_ok(self):
        code = 'secret = os.getenv("SECRET_KEY")\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) == 0

    def test_comment_ignored(self):
        code = '# password = "example"\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) == 0

    def test_clean_code(self):
        code = "x = 1\ny = 2\nresult = x + y\n"
        findings = self.detector.detect(code, "math.py")
        assert len(findings) == 0

    def test_has_remediation(self):
        code = 'password = "hunter2"\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) >= 1
        assert all(f.remediation for f in findings)

    def test_severity_is_critical(self):
        code = 'api_key = "sk-abc123def456ghi789jklmnopqrstuvwxyz"\n'
        findings = self.detector.detect(code, "config.py")
        assert any(f.severity == "critical" for f in findings)


# --- SQL Injection ---


class TestSQLInjectionDetector:
    def setup_method(self):
        self.detector = SQLInjectionDetector()

    def test_fstring_select(self):
        code = 'query = f"SELECT * FROM users WHERE id = {user_id}"\n'
        findings = self.detector.detect(code, "db.py")
        assert len(findings) >= 1
        assert any("f-string" in f.description for f in findings)

    def test_fstring_insert(self):
        code = 'query = f"INSERT INTO logs (msg) VALUES ({message})"\n'
        findings = self.detector.detect(code, "db.py")
        assert len(findings) >= 1

    def test_format_select(self):
        code = 'query = "SELECT * FROM users WHERE name = {}".format(name)\n'
        findings = self.detector.detect(code, "db.py")
        assert len(findings) >= 1
        assert any(".format()" in f.description for f in findings)

    def test_concatenation_in_execute(self):
        code = 'cursor.execute("SELECT * FROM users WHERE id = " + user_id)\n'
        findings = self.detector.detect(code, "db.py")
        assert len(findings) >= 1
        assert any("concatenation" in f.description for f in findings)

    def test_parameterized_ok(self):
        code = 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))\n'
        findings = self.detector.detect(code, "db.py")
        assert len(findings) == 0

    def test_static_query_ok(self):
        code = 'query = "SELECT * FROM users WHERE active = 1"\n'
        findings = self.detector.detect(code, "db.py")
        assert len(findings) == 0

    def test_clean_code(self):
        code = "def add(a, b):\n    return a + b\n"
        findings = self.detector.detect(code, "math.py")
        assert len(findings) == 0

    def test_has_remediation(self):
        code = 'query = f"SELECT * FROM users WHERE id = {uid}"\n'
        findings = self.detector.detect(code, "db.py")
        assert all(f.remediation for f in findings)

    def test_syntax_error_handled(self):
        code = "def foo(:\n"
        findings = self.detector.detect(code, "bad.py")
        # Should not crash
        assert isinstance(findings, list)


# --- Shell Injection ---


class TestShellInjectionDetector:
    def setup_method(self):
        self.detector = ShellInjectionDetector()

    def test_os_system_fstring(self):
        code = 'os.system(f"rm -rf {path}")\n'
        findings = self.detector.detect(code, "cleanup.py")
        assert len(findings) >= 1
        assert any("os.system" in f.description for f in findings)

    def test_os_system_concatenation(self):
        code = 'os.system("ls " + directory)\n'
        findings = self.detector.detect(code, "list.py")
        assert len(findings) >= 1

    def test_subprocess_shell_true(self):
        code = 'subprocess.run(f"echo {msg}", shell=True)\n'
        findings = self.detector.detect(code, "run.py")
        assert len(findings) >= 1
        assert any("shell=True" in f.description for f in findings)

    def test_subprocess_popen_shell_true(self):
        code = "subprocess.Popen(cmd, shell=True)\n"
        findings = self.detector.detect(code, "run.py")
        assert len(findings) >= 1

    def test_os_popen(self):
        code = 'os.popen(f"cat {filename}")\n'
        findings = self.detector.detect(code, "read.py")
        assert len(findings) >= 1

    def test_subprocess_list_ok(self):
        code = 'subprocess.run(["ls", "-l", path], check=True)\n'
        findings = self.detector.detect(code, "list.py")
        assert len(findings) == 0

    def test_os_system_static_ok(self):
        code = 'os.system("clear")\n'
        findings = self.detector.detect(code, "ui.py")
        assert len(findings) == 0

    def test_subprocess_no_shell_ok(self):
        code = 'subprocess.run(["git", "status"], check=True)\n'
        findings = self.detector.detect(code, "git.py")
        assert len(findings) == 0

    def test_clean_code(self):
        code = "x = 1\n"
        findings = self.detector.detect(code, "simple.py")
        assert len(findings) == 0

    def test_severity_is_critical(self):
        code = 'os.system(f"rm {path}")\n'
        findings = self.detector.detect(code, "cleanup.py")
        assert all(f.severity == "critical" for f in findings)

    def test_syntax_error_handled(self):
        code = "def foo(:\n"
        findings = self.detector.detect(code, "bad.py")
        assert findings == []


# --- File Ops ---


class TestFileOpsDetector:
    def setup_method(self):
        self.detector = FileOpsDetector()

    def test_open_fstring_path(self):
        code = 'f = open(f"/data/{user_file}")\n'
        findings = self.detector.detect(code, "read.py")
        assert len(findings) >= 1
        assert any("open()" in f.description for f in findings)

    def test_open_concatenation(self):
        code = 'f = open(base_dir + "/" + filename)\n'
        findings = self.detector.detect(code, "read.py")
        assert len(findings) >= 1

    def test_open_variable(self):
        code = "f = open(user_path)\n"
        findings = self.detector.detect(code, "read.py")
        assert len(findings) >= 1

    def test_path_traversal(self):
        code = 'path = "../../etc/passwd"\n'
        findings = self.detector.detect(code, "read.py")
        assert len(findings) >= 1
        assert any("traversal" in f.description.lower() for f in findings)

    def test_sensitive_path(self):
        code = 'data = open("/etc/passwd").read()\n'
        findings = self.detector.detect(code, "read.py")
        assert len(findings) >= 1
        assert any("sensitive" in f.description.lower() for f in findings)

    def test_ssh_path(self):
        code = 'key = open("~/.ssh/id_rsa").read()\n'
        findings = self.detector.detect(code, "ssh.py")
        assert len(findings) >= 1

    def test_env_file(self):
        code = 'config = open(".env").read()\n'
        findings = self.detector.detect(code, "config.py")
        assert len(findings) >= 1

    def test_open_literal_ok(self):
        code = 'f = open("config.json")\n'
        findings = self.detector.detect(code, "read.py")
        assert len(findings) == 0

    def test_clean_code(self):
        code = "x = 1\nprint(x)\n"
        findings = self.detector.detect(code, "simple.py")
        assert len(findings) == 0

    def test_has_remediation(self):
        code = 'f = open(f"/data/{name}")\n'
        findings = self.detector.detect(code, "read.py")
        assert all(f.remediation for f in findings)

    def test_syntax_error_handled(self):
        code = "def foo(:\n"
        findings = self.detector.detect(code, "bad.py")
        assert isinstance(findings, list)


# --- Dependencies ---


class TestDependenciesDetector:
    def setup_method(self):
        self.detector = DependenciesDetector()

    def test_unpinned_pip_install(self):
        code = "# pip install requests\n"
        # Comments are skipped
        findings = self.detector.detect(code, "setup.py")
        assert len(findings) == 0

    def test_unpinned_in_code(self):
        code = 'os.system("pip install requests")\n'
        findings = self.detector.detect(code, "setup.py")
        assert any(
            "unpinned" in f.description.lower() or "subprocess" in f.description.lower()
            for f in findings
        )

    def test_url_install(self):
        code = 'os.system("pip install https://example.com/evil.tar.gz")\n'
        findings = self.detector.detect(code, "setup.py")
        assert len(findings) >= 1
        assert any("URL" in f.description for f in findings)

    def test_git_install(self):
        code = 'os.system("pip install git+https://github.com/user/repo")\n'
        findings = self.detector.detect(code, "setup.py")
        assert len(findings) >= 1

    def test_subprocess_pip(self):
        code = 'subprocess.run("pip install foo", shell=True)\n'
        findings = self.detector.detect(code, "install.py")
        assert len(findings) >= 1

    def test_dynamic_import(self):
        code = "__import__(module_name)\n"
        findings = self.detector.detect(code, "loader.py")
        assert len(findings) >= 1
        assert any("__import__" in f.description for f in findings)

    def test_static_import_ok(self):
        code = '__import__("json")\n'
        findings = self.detector.detect(code, "loader.py")
        assert len(findings) == 0

    def test_normal_import_ok(self):
        code = "import json\nfrom pathlib import Path\n"
        findings = self.detector.detect(code, "app.py")
        assert len(findings) == 0

    def test_pinned_install_comment(self):
        # This is a comment, so it's skipped
        code = "# pip install requests==2.31.0\n"
        findings = self.detector.detect(code, "setup.py")
        assert len(findings) == 0

    def test_clean_code(self):
        code = "x = 1\n"
        findings = self.detector.detect(code, "simple.py")
        assert len(findings) == 0


# --- Data Exposure ---


class TestDataExposureDetector:
    def setup_method(self):
        self.detector = DataExposureDetector()

    def test_ssn_in_print(self):
        code = 'print(f"SSN: 123-45-6789")\n'
        findings = self.detector.detect(code, "user.py")
        assert len(findings) >= 1
        assert any("SSN" in f.description for f in findings)

    def test_password_in_log(self):
        code = 'logger.info(f"User password: {password}")\n'
        findings = self.detector.detect(code, "auth.py")
        assert len(findings) >= 1
        assert any("sensitive" in f.description.lower() for f in findings)

    def test_secret_in_error(self):
        code = 'raise ValueError(f"Invalid secret: {secret}")\n'
        findings = self.detector.detect(code, "auth.py")
        assert len(findings) >= 1

    def test_token_in_print(self):
        code = 'print(f"Token: {auth_token}")\n'
        findings = self.detector.detect(code, "debug.py")
        assert len(findings) >= 1

    def test_normal_log_ok(self):
        code = 'logger.info("User logged in successfully")\n'
        findings = self.detector.detect(code, "auth.py")
        assert len(findings) == 0

    def test_normal_print_ok(self):
        code = 'print("Hello, world!")\n'
        findings = self.detector.detect(code, "app.py")
        assert len(findings) == 0

    def test_comment_ignored(self):
        code = '# print(f"password: {pwd}")\n'
        findings = self.detector.detect(code, "auth.py")
        assert len(findings) == 0

    def test_clean_code(self):
        code = "result = compute()\nprint(result)\n"
        findings = self.detector.detect(code, "app.py")
        assert len(findings) == 0

    def test_has_remediation(self):
        code = 'logger.info(f"password reset for {password}")\n'
        findings = self.detector.detect(code, "auth.py")
        assert all(f.remediation for f in findings)


# --- Integration ---


class TestScanAll:
    def test_clean_code_no_findings(self):
        code = "def add(a, b):\n    return a + b\n"
        findings = scan_all(code, "math.py")
        assert len(findings) == 0

    def test_multiple_issues(self):
        code = (
            'password = "hunter2"\n'
            'query = f"SELECT * FROM users WHERE id = {uid}"\n'
            'os.system(f"rm {path}")\n'
        )
        findings = scan_all(code, "bad.py")
        patterns = {f.pattern for f in findings}
        assert "hardcoded_credentials" in patterns
        assert "sql_injection" in patterns
        assert "shell_injection" in patterns

    def test_findings_have_line_numbers(self):
        code = 'x = 1\npassword = "secret"\ny = 2\n'
        findings = scan_all(code, "config.py")
        assert all(f.line_number > 0 for f in findings)
        assert any(f.line_number == 2 for f in findings)

    def test_findings_have_content(self):
        code = 'api_key = "sk-abc123def456ghi789jklmnopqrstuvwxyz"\n'
        findings = scan_all(code, "config.py")
        assert all(f.line_content for f in findings)
        assert all(f.description for f in findings)
        assert all(f.remediation for f in findings)
