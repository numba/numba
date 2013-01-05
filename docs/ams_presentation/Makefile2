latex:
	pandoc -S -t beamer coco.md --template beamer.template -V theme:Warsaw --latex-engine=xelatex -o coco.pdf

html:
	pandoc --self-contained --data-dir=. -s -t slidy -i coco.md -o coco.html
