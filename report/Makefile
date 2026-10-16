MAIN = main
LATEX = pdflatex

.PHONY: all clean

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex literature.bib IEEEtran.cls IEEEtran.bst
	$(LATEX) $(MAIN)
	bibtex $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f $(MAIN).pdf *.aux *.bbl *.blg *.log *.out
