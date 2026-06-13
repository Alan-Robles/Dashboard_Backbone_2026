# gunicorn.conf.py
# Timeout extendido para aguantar la carga inicial de datos (CSVs + preprocesamiento)
timeout = 120         # segundos antes de matar un worker que no responde
workers = 1           # un solo worker para respetar el límite de 512 MB
worker_class = "sync"
bind = "0.0.0.0:10000"
