import sys
import logging
from pathlib import Path
from application import create_app


sys.path.append(
    list(Path(__file__).absolute().parents)[1].as_posix()
)
app= create_app()


if __name__!="__main__":
    logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = logger.handlers
    app.logger.setLevel(logger.level)

if __name__=="__main__":
    app.run()
