import logging


def configure_logging():
  logging.basicConfig(level=logging.DEBUG)
  logging.getLogger('tensorflow').setLevel(logging.ERROR)
  logging.getLogger('matplotlib').setLevel(logging.ERROR)
