from fastapi import FastAPI

app = FastAPI(title="My API")

@app.get("/health")
def health():
    return {"status": "ok"}
