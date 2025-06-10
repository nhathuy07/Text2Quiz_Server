redis-server --save "" &
uvicorn main:app --port 4200 --reload
