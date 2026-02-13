"""Critical pattern detectors for the security gate.

Import this module to register all detectors.
"""

from .base import PatternDetector, get_all_detectors, register_detector, scan_all
from .credentials import CredentialsDetector
from .data_exposure import DataExposureDetector
from .dependencies import DependenciesDetector
from .file_ops import FileOpsDetector
from .shell_injection import ShellInjectionDetector
from .sql_injection import SQLInjectionDetector

__all__ = [
    "CredentialsDetector",
    "DataExposureDetector",
    "DependenciesDetector",
    "FileOpsDetector",
    "PatternDetector",
    "SQLInjectionDetector",
    "ShellInjectionDetector",
    "get_all_detectors",
    "register_detector",
    "scan_all",
]
