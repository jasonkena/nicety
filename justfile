default:
    just --list

test:
    uv run -m unittest tests.unit_test
