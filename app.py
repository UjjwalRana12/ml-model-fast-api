from fastapi import FastAPI
from cancer import cancer_router
from heart import heart_router
from diabetes import diabetes_router

app=FastAPI()

app.include_router(cancer_router)
app.include_router(diabetes_router)
app.include_router(heart_router)