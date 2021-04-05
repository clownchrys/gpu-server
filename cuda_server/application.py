from flask import Flask


def create_app(**config_overrides):
    app = Flask(__name__)
    app.config.from_pyfile('settings.py')
    app.config.update(config_overrides)

    # import applications
    from apps import main

    # register applications
    app.register_blueprint(main.app)

    return app
