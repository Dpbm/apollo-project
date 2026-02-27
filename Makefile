DASHBOARD := dashboard.py
REQUIREMENTS := requirements.txt


all: $(REQUIREMENTS)

.PHONY: run
run: $(DASHBOARD)
	uv run streamlit run $(DASHBOARD)


$(REQUIREMENTS):
	uv pip freeze > $(REQUIREMENTS)
