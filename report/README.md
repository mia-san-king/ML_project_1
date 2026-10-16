# Project 1 report (LaTeX)

This folder follows the CS-433 Project 1 template:

[github.com/epfml/ML_course/tree/main/projects/project1/latex-example-paper](https://github.com/epfml/ML_course/tree/main/projects/project1/latex-example-paper)

| Course template | This repo |
| --- | --- |
| `latex-template.tex` | `main.tex` (MICHD content) |
| `literature.bib` | `literature.bib` |
| `IEEEtran.cls`, `IEEEtran.bst` | same files (copied) |

Build (same commands as the template, with `main` instead of `groupXX-submission`):

```bash
cd report
make
# or: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Output: `main.pdf` (submit this PDF on [mlcourse.epfl.ch](http://mlcourse.epfl.ch)).
