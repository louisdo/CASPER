.PHONY: venv-dense venv-colbert venv-cspr lock help

venv-dense:
	UV_PROJECT_ENVIRONMENT=.venv-dense uv sync --group dense

venv-colbert:
	UV_PROJECT_ENVIRONMENT=.venv-colbert uv sync --group colbert

venv-cspr:
	UV_PROJECT_ENVIRONMENT=.venv-cspr uv sync --group cspr

lock:
	uv lock

help:
	@echo "make venv-dense    — create/update .venv-dense with dense retrieval deps"
	@echo "make venv-colbert  — create/update .venv-colbert with ColBERT deps"
	@echo "make venv-cspr     — create/update .venv-cspr with CSpR deps (dense + cspr-specific)"
	@echo "make lock          — regenerate uv.lock"
