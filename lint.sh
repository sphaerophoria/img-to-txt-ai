#!/usr/bin/env bash

set -ex

function python_files() {
	git ls-files -z "*.py"
}

function js_files() {
	git ls-files "*.js" | grep -v "^\."
}

function html_files() {
	git ls-files -z "*.html"
}

function css_files() {
	git ls-files -z "*.css"
}

python_files | xargs -0 black --check
python_files | xargs -0 ruff
css_files | xargs -0 prettier -c
css_files | xargs -0 stylelint
html_files | xargs -0 htmlhint
html_files | xargs -0 prettier -c
js_files | xargs prettier -c
js_files | xargs eslint
