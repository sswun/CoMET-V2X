# Manuscript

**CoMET: Tracking the Shared Pose Error of Roadside Messages for Asynchronous Vehicle–Infrastructure Cooperative 3D Detection**

Shuwei Sun, Jian Huang and Hui Cao · Xi'an Jiaotong University

[Read the manuscript (PDF)](main.pdf) · [Permission to use the code, figures and results](../LICENSE)

| Path | Contents |
|---|---|
| `main.tex`, `sec/` | LaTeX source; `sec/A_appendix.tex` is the appendix |
| `references.bib` | bibliography; `main.bbl` is kept for arXiv upload |
| `tables/` | table bodies generated from the result files |
| `fig/` | manuscript figures, including the illustrated method and validation trace in `overview.pdf` |
| `numbers.json` | extracted statistics used in the manuscript tables and figures |
| `scripts/` | extract statistics and generate the LaTeX tables and figures |
| `main.pdf` | compiled manuscript |

After receiving permission, run these commands from `paper/` to regenerate the manuscript from the released results. The scripts use NumPy and Matplotlib. The PDF build uses `latexmk` and a LaTeX installation with the packages listed in `main.tex`.

```bash
python3 scripts/extract_numbers.py
python3 scripts/make_tables.py
python3 scripts/make_figures.py
python3 scripts/make_fig_overview.py
latexmk -pdf main.tex
```

Numbers written in the running text were checked by hand against `numbers.json` and `../docs/FINDINGS.md`. If a result file changes, regenerate the tables and figures and re-check the text.

Not included: the training-budget dose-response of the second tier.
