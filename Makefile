.PHONY: test demo eval api

test:
	pytest tests/ -q

demo:
	python -m scripts.demo

eval:
	python -m eval.run_eval

api:
	uvicorn expense_agent.api:app --reload
