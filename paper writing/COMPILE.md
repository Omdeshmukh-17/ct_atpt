# Compiling `paper_ctatpt.tex`

## Files

| File | Purpose |
|---|---|
| `paper_ctatpt.tex` | Main paper (IEEEtran journal class) |
| `paper_ctatpt.bib` | Bibliography database (BibTeX) |
| `paper_ctatpt_draft.md` | Same paper as Markdown for quick reading / editing |
| `figures/` | Place figure PDFs here (see Figures section below) |

## Local compile

You need a LaTeX distribution with `IEEEtran.cls` and `IEEEtran.bst`. Most
TeX Live / MiKTeX installations include them.

```bash
cd /home/aries/paper
pdflatex paper_ctatpt
bibtex   paper_ctatpt
pdflatex paper_ctatpt
pdflatex paper_ctatpt
```

The four-pass sequence resolves citations and cross-references. `latexmk` is
the lazy alternative:

```bash
latexmk -pdf paper_ctatpt.tex
```

## Overleaf

1. Create a new project, upload `paper_ctatpt.tex` and `paper_ctatpt.bib`.
2. Set the compiler to **pdfLaTeX** (Menu → Compiler).
3. Set the main document to `paper_ctatpt.tex`.
4. Hit **Recompile**.

## Result placeholders

Every number that depends on training results is wrapped in `\R{label}` and
renders as `[label]` in red. Search for `\R{` in the source to find all
remaining placeholders. Replace with the actual number once training
finishes:

```latex
% before
\R{AUC}

% after (example)
$0.913$
```

## Figures

The TeX expects three PDFs in `figures/`:

| Path | Description |
|---|---|
| `figures/architecture.pdf` | The pipeline diagram (`ct_atpt_architecture.excalidraw` in the research repo — export to PDF from Excalidraw) |
| `figures/trajectories.pdf` | $\alpha, \beta, \gamma, \lambda$ trajectories over training (matplotlib export) |
| `figures/token_maps.pdf` | Token-retention maps on representative cases |

To suppress missing-figure errors during early compiles, comment out the
three `\includegraphics{}` lines or wrap them in `\IfFileExists`.

## Targeting different IEEE journals

The class file is the same (`IEEEtran` with the `journal` option) for:
- IEEE Transactions on Medical Imaging (TMI)
- IEEE Journal of Biomedical and Health Informatics (JBHI)
- IEEE Access (use `\documentclass[journal]{IEEEtran}` — they also accept
  `\documentclass[journal,final]{IEEEtran}`)

For IEEE Access, also set the page-style banner per their template; the
content does not change.

## Pre-submission checklist

- [ ] Replace every `\R{...}` and `\todo{...}` marker
- [ ] Add real author names, affiliations, and emails
- [ ] Insert the three figure PDFs
- [ ] Complete the `[PLACEHOLDER]` entries in `paper_ctatpt.bib`
- [ ] Run `pdflatex` twice + `bibtex` + `pdflatex` for stable references
- [ ] Read the compiled PDF top-to-bottom; nothing should still be red
- [ ] Submit
