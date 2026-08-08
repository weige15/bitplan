.PHONY: m1-smoke test validate

m1-smoke:
	python3 -m bitplan.smoke --config configs/m1.json --output results/raw/m1-smoke-fixture-v1

test:
	python3 -m unittest discover -s tests -v

validate:
	python3 scripts/validate_research_data.py
