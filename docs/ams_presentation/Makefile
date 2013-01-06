html:
	pandoc --self-contained --data-dir=. -s -t slidy -i \
		why.md compare.md example_simpleadd.md \
		example_mandel.md table.md features.md -o main.html

latex:
	pandoc -S -t beamer coco.md --template beamer.template -V theme:Warsaw --latex-engine=xelatex -o coco.pdf


