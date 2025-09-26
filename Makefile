clean:
	@echo "Cleaning logs"
	rm -r logs/*

black:
	black -l 88 -t py39 ./src --exclude context-cite --exclude outlines
	black -l 88 -t py39 ./scripts