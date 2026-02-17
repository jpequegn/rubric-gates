"""Critical pattern detectors for the security gate.

Import this module to register all detectors.
"""

from .base import (
    PatternDetector,
    get_all_detectors,
    get_configured_detectors,
    register_detector,
    scan_all,
)
from .credentials import CredentialsDetector
from .custom import CustomPatternDetector, load_custom_detectors
from .data_exposure import DataExposureDetector
from .dependencies import DependenciesDetector
from .file_ops import FileOpsDetector
from .shell_injection import ShellInjectionDetector
from .sql_injection import SQLInjectionDetector

__all__ = [
    "CredentialsDetector",
    "CustomPatternDetector",
    "DataExposureDetector",
    "DependenciesDetector",
    "FileOpsDetector",
    "PatternDetector",
    "SQLInjectionDetector",
    "ShellInjectionDetector",
    "get_all_detectors",
    "get_configured_detectors",
    "load_custom_detectors",
    "register_detector",
    "scan_all",
]
