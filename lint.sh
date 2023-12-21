#!/usr/bin/env bash

set -ex

function python_files() {
	git ls-files -z "*.py"
}

python_files | xargs -0 black --check
python_files | xargs -0 ruff
