"""Configuration for pytest."""

import socket

import pytest


def has_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """Check if there is an internet connection available.

    Args:
        host: The host to try to connect to (default: Google DNS)
        port: The port to try to connect to (default: DNS port)
        timeout: Connection timeout in seconds

    Returns:
        bool: True if internet connection is available, False otherwise
    """
    socket.setdefaulttimeout(timeout)
    try:
        # Use a context manager to ensure the socket is properly closed
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            return True
    except (OSError, socket.timeout):
        return False


# Create a fixture to skip network tests when offline
def pytest_addoption(parser):
    parser.addoption(
        "--skip-network",
        action="store_true",
        default=False,
        help="Skip tests that require network access",
    )
    parser.addoption(
        "--force-network",
        action="store_true",
        default=False,
        help="Force network tests even if no connection is detected",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "network: mark test as requiring network access")


def pytest_collection_modifyitems(config, items):
    skip_network = config.getoption("--skip-network")
    force_network = config.getoption("--force-network")

    # Auto-detect network connectivity if not explicitly skipped and not forced
    if not skip_network and not force_network:
        skip_network = not has_internet_connection()

    if skip_network:
        reason = (
            "Network tests disabled (no connection detected or --skip-network used)"
        )
        skip_marker = pytest.mark.skip(reason=reason)
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_marker)
