import uvicorn
from fastapi import FastAPI

from cjw.knowledgeqa.api.Knowledge import Knowledge

app = FastAPI()


@app.get("/knowledge")
async def ask():
    return {"message": "Hello there"}


if __name__ == '__main__':
    knowledge = Knowledge()
    uvicorn.run(app, host="0.0.0.0", port=8000)
