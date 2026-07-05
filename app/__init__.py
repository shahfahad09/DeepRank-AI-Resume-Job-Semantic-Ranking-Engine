import os
from flask import Flask

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def create_app():
    app = Flask(__name__)
    app.config["BASE_DIR"] = BASE_DIR
    app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploads", "resumes")
    app.config["OUTPUT_FOLDER"] = os.path.join(BASE_DIR, "outputs", "sorted_resumes")
    app.config["EXPORT_FOLDER"] = os.path.join(BASE_DIR, "exports")
    app.config["DB_PATH"] = os.path.join(BASE_DIR, "instance", "ranking_history.db")
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024

    for folder in [
        app.config["UPLOAD_FOLDER"],
        app.config["OUTPUT_FOLDER"],
        app.config["EXPORT_FOLDER"],
        os.path.dirname(app.config["DB_PATH"])
    ]:
        os.makedirs(folder, exist_ok=True)

    from .database import init_db
    init_db(app.config["DB_PATH"])

    from .routes import main
    app.register_blueprint(main)

    return app
