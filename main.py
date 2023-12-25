# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modeltest import search
# Import hàm predict từ simple_model.py
from fastapi.middleware.cors import CORSMiddleware
origins = ["http://localhost:3000"]  # Add the origin of your front-end application

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Define Pydantic models for input and output
class InputData(BaseModel):
    string: str
    k: int


class OutputData(BaseModel):
    output_value: list

# API endpoint to process input and return output
@app.post("/predict", response_model=OutputData)
async def predict_endpoint(input_data: InputData):
    try:
        if(input_data.string == ""):
            output_data = OutputData(output_value=[])
            return output_data
        else:
        # Gọi hàm search từ mô hình đã tải
            result = search(input_data.string, input_data.k )
            output_data = OutputData(output_value=result)
            return output_data #import hàm search vào
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server with Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
