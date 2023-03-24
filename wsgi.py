#!/usr/bin/env python

from callbacks import app

# entry point for Dockerfile -> gunicorn
server = app.server

if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    # testing
    app.run_server(debug=True)
    # production
    # app.run_server(debug=False, host="0.0.0.0", port="8050")
