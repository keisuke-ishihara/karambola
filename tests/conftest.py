"""Shared test fixtures."""

import os
import pytest

# Path to test inputs
TEST_INPUTS = os.path.join(os.path.dirname(__file__), "..", "test_suite", "inputs")


@pytest.fixture
def test_inputs_dir():
    return TEST_INPUTS
