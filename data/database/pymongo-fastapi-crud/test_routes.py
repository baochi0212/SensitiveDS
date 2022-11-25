from fastapi import APIRouter, Body, Request, Response, HTTPException, status
from fastapi.encoders import jsonable_encoder
from typing import List

from models import Student

router = APIRouter()
@router.post("/", response_description="Add new student", status_code=status.HTTP_201_CREATED, response_model=Student)
def post_item(request: Request, student: Student = Body(...)):
    student = jsonable_encoder(student)
    new_student = request.app.database['testcollection'].insert_one(student)
    created_student = request.app.database["testcollection"].find_one(
        {"_id": new_student.inserted_id}
    )
    

    return created_student

@router.get("/", response_description="List all students", response_model=List[Student])
def list_books(request: Request):
    books = list(request.app.database["testcollection"].find(limit=100))
    return books