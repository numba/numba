html:
	pandoc --css table.css --self-contained --data-dir=. -s -t slidy -i \
		title.md why.md compare.md example_simpleadd.md \
		example_mandel.md features.md numbapro.md dynamic.md \
		future.md end.md -o main.html

latex:
	pandoc -S -t beamer coco.md --template beamer.template -V theme:Warsaw --latex-engine=xelatex -o coco.pdf


