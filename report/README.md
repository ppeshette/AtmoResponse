# Report

The report documents live under `report/`:

- `PROJECT_SUMMARY.md` and `PROJECT_SUMMARY.pdf`: the project summary, results, and acquisition
  recommendation. Three pages of main text plus a references page.
- `METHOD_NOTES.md`: additional algorithm results and the reconstruction limits that qualify every
  Realized Sensitivity figure.
- `ACQUISITION_TARGETS.md`: the ranked candidate set and scoring method behind the summary's short
  list of ten targets.
- `FIGURES.md` and `FIGURES.pdf`: captions and a rendered sheet for the four notebook figures.

The two PDFs are generated from the Markdown and the figure PNGs by `render_pdf.py` and
`render_figures_pdf.py` (`pip install "atmoresponse[report]"`).

The four case-study figure values are in `src/atmoresponse/assets/figure_values.json`, which the
annotated notebook writes. The Method Notes tables come from the fuller analysis behind the summary.
