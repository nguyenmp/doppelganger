
run : test ;
	FLASK_APP=doppelganger.flask_app FLASK_ENV=development flask run

test : ;
	pylint doppelganger tests/*.py
	python -m pytest --cov=doppelganger tests/
