from os import path

ROOT_DIR = path.realpath(path.join(path.dirname(__file__), '..'))
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
LOG_DIR = path.join(ROOT_DIR, "logs")
EXP_LOG_DIR = path.join(LOG_DIR, "experiments")