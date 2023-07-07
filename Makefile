format:
	python run black linprog/.
	python run black tests/.
	python run isort linprog/.
	python run isort tests/.
	
lint: 
	poetry run ruff check linprog/. --fix
	poetry run ruff check tests/. --fix

test:
	poetry run pytest tests/.