format:
	poetry run black linprog/
	poetry run isort linprog/
	
	poetry run black tests/
	poetry run isort tests/
	
lint: 
	poetry run ruff check linprog/. --fix
	poetry run ruff check tests/. --fix

test:
	poetry run pytest tests/.