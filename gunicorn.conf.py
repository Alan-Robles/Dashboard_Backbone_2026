# gunicorn.conf.py
timeout = 120
workers = 1
worker_class = "sync"

def post_fork(server, worker):
    """Se ejecuta en el worker DESPUÉS de que el puerto ya está ligado."""
    from dashboard2026 import load_data
    load_data()
