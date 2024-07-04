import pytest

def sum(a, b):
    return a + b

def test_all(benchmark):
    benchmark(sum, 1, 2)
