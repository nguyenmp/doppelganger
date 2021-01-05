# doppelganger
Given a face, find the most similar faces out of a known collection

It's currently set up to query a specific LDAP server for profile pictures.

There's an active branch that uses k-means clustering and approximate nearest neighbors but is currently not pushed due to privacy concerns.

## Usage

You can run as a standalone CLI for one-off queries and to build the database:

```
pip install -e .
python doppelganger init
python doppelganger analyze your_person_id_from_ldap
```

You can alternatively run as a webserver once the database is set up:

```
pip install -e .
python doppelganger init
FLASK_APP=doppelganger.flask_app python -m flask run --host=0.0.0.0 --port=80 >> log.stdout 2>> log.stderr &
```
