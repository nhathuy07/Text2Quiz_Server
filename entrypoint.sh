redis-server --save "" &
uvicorn main:app --port 7860 --host 0.0.0.0 --workers 5