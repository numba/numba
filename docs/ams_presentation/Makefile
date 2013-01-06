latex:
	pandoc -S -t beamer coco.md --template beamer.template -V theme:Warsaw --latex-engine=xelatex -o coco.pdf

html:
	pandoc --self-contained --data-dir=. -s -t slidy -i why.rst compare.rst example_simpleadd.rst example_mandel.rst features.rst -o main.html
