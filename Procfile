web: gunicorn --bind 0.0.0.0:$PORT app:app --worker-class uvicorn.workers.UvicornWorker --workers 1 --timeout 300 --preload
worker: python worker.py