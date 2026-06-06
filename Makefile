.PHONY: venv-dense venv-colbert lock help

venv-dense:
	UV_PROJECT_ENVIRONMENT=.venv-dense uv sync --group dense

venv-colbert:
	UV_PROJECT_ENVIRONMENT=.venv-colbert uv sync --group colbert

lock:
	uv lock

help:
	@echo "make venv-dense    — create/update .venv-dense with dense retrieval deps"
	@echo "make venv-colbert  — create/update .venv-colbert with ColBERT deps"
	@echo "make lock          — regenerate uv.lock"
