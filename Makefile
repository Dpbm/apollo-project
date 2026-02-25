DASHBOARD := dashboard.py

run: $(DASHBOARD)
	uv run streamlit run $(DASHBOARD)
