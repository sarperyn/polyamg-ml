PYTHON ?= .venv/bin/python
PYTHONPATH ?= py_ml
MPLCONFIGDIR ?= /private/tmp/polyamg-ml-matplotlib
SOLVER_TIMES_VERTICAL_STRIPES4_OUT ?= data/reports/theta_solver_times/vertical_stripes4_solver_times.png
SOLVER_TIMES_CHECKER4X4_OUT ?= data/reports/theta_solver_times/checker4x4_solver_times.png
TIME_VS_RHO_OUT ?= data/reports/time_vs_rho/time_vs_rho.png
THETA_LEVELS_VERTICAL_SPLIT_OUT ?= data/reports/theta_levels_relation/vertical_split_theta_vs_levels.png
THETA_LEVELS_VERTICAL_STRIPES4_OUT ?= data/reports/theta_levels_relation/vertical_stripes4_theta_vs_levels.png
THETA_LEVELS_CHECKER2X2_OUT ?= data/reports/theta_levels_relation/checker2x2_theta_vs_levels.png
THETA_LEVELS_CHECKER4X4_OUT ?= data/reports/theta_levels_relation/checker4x4_theta_vs_levels.png
THETA_RHO_REPORT_DIR ?= data/reports/theta_rho_relation_all_patterns
BASELINE_VERTICAL_SPLIT_GLOB ?= data/raw/baseline_vertical_stripes/*.json
BASELINE_VERTICAL_STRIPES4_GLOB ?= data/raw/baseline_vertical_stripes4/*.json
BASELINE_CHECKER2X2_GLOB ?= data/raw/baseline_checker2x2/*.json
BASELINE_CHECKER4X4_GLOB ?= data/raw/baseline_checker4x4/*.json
BASELINE_ALL_GLOB ?= data/raw/baseline_*/*.json

.PHONY: reports report-figures theta-levels-figures theta-levels-vertical-split theta-levels-vertical-stripes4 theta-levels-checker2x2 theta-levels-checker4x4 solver-times-figures solver-times-vertical-stripes4 solver-times-checker4x4 time-vs-rho-figure figure3-solver-times reproduce-figure3 theta-levels-figure theta-rho-tables
reports: report-figures theta-rho-tables

report-figures: theta-levels-figures solver-times-figures time-vs-rho-figure

solver-times-figures: solver-times-vertical-stripes4 solver-times-checker4x4

theta-levels-figures: theta-levels-vertical-split theta-levels-vertical-stripes4 theta-levels-checker2x2 theta-levels-checker4x4

theta-levels-vertical-split:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/plot_theta_levels.py \
		--input_glob '$(BASELINE_VERTICAL_SPLIT_GLOB)' \
		--out $(THETA_LEVELS_VERTICAL_SPLIT_OUT)

theta-levels-vertical-stripes4:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/plot_theta_levels.py \
		--input_glob '$(BASELINE_VERTICAL_STRIPES4_GLOB)' \
		--out $(THETA_LEVELS_VERTICAL_STRIPES4_OUT)

theta-levels-checker2x2:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/plot_theta_levels.py \
		--input_glob '$(BASELINE_CHECKER2X2_GLOB)' \
		--out $(THETA_LEVELS_CHECKER2X2_OUT)

theta-levels-checker4x4:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/plot_theta_levels.py \
		--input_glob '$(BASELINE_CHECKER4X4_GLOB)' \
		--out $(THETA_LEVELS_CHECKER4X4_OUT)

theta-levels-figure: theta-levels-figures

solver-times-vertical-stripes4:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/plot_theta_solver_times.py \
		--input_glob '$(BASELINE_VERTICAL_STRIPES4_GLOB)' \
		--out $(SOLVER_TIMES_VERTICAL_STRIPES4_OUT)

solver-times-checker4x4:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/plot_theta_solver_times.py \
		--input_glob '$(BASELINE_CHECKER4X4_GLOB)' \
		--out $(SOLVER_TIMES_CHECKER4X4_OUT)

time-vs-rho-figure:
	MPLCONFIGDIR=$(MPLCONFIGDIR) PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/plot_time_vs_rho.py \
		--input_glob '$(BASELINE_VERTICAL_SPLIT_GLOB)' \
		--input_glob '$(BASELINE_VERTICAL_STRIPES4_GLOB)' \
		--input_glob '$(BASELINE_CHECKER2X2_GLOB)' \
		--input_glob '$(BASELINE_CHECKER4X4_GLOB)' \
		--out $(TIME_VS_RHO_OUT)

figure3-solver-times: solver-times-figures

reproduce-figure3: solver-times-figures

theta-rho-tables:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/generate_theta_rho_tables.py \
		--vertical_split_glob '$(BASELINE_VERTICAL_SPLIT_GLOB)' \
		--stripes_glob '$(BASELINE_VERTICAL_STRIPES4_GLOB)' \
		--checker2x2_glob '$(BASELINE_CHECKER2X2_GLOB)' \
		--checker_glob '$(BASELINE_CHECKER4X4_GLOB)' \
		--out_dir $(THETA_RHO_REPORT_DIR)
	@if command -v pdflatex >/dev/null 2>&1; then \
		cd $(THETA_RHO_REPORT_DIR) && pdflatex -interaction=nonstopmode theta_rho_tables.tex >/dev/null; \
	else \
		echo "pdflatex not found; generated theta_rho_tables.tex and CSV files only."; \
	fi
