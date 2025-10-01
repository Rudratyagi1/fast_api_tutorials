# import necessary libraries and modules
from fastapi.responses import JSONResponse
from fastapi import FastAPI,Path,HTTPException, Query
import json
from pydantic import BaseModel, Field ,computed_field
from typing import Annotated,Literal, Optional


# create an instance of FastAPI
app = FastAPI()


# define patient pydantic model
class Patient(BaseModel):
    id: Annotated[str , Field(...,description="Unique identifier for the patient", examples=["P001"])] 
    name: Annotated[str , Field(...,description="Name of the patient", examples=["John Doe"])]
    city : Annotated[str , Field(...,description="City of the patient", examples=["New York"])]
    age: Annotated[int , Field(...,gt=0,lt=120,description="Age of the patient", examples=[30])]
    gender : Annotated[Literal['male','female','others'], Field(...,description = "gender of the patient", examples=["M"])]
    height: Annotated[float , Field(...,gt=0,description="Height of the patient in mtrs", examples=[175.5])]
    weight: Annotated[float , Field(...,gt=0,description="Weight of the patient in kg", examples=[70.5])]
    
    @computed_field
    @property
    def bmi(self) -> float:
        bmi =  round(self.weight / (self.height ** 2), 2)
        return bmi
    
    @computed_field
    @property
    def verdict(self) -> str:
        bmi = self.bmi
        if bmi < 18.5:
            return "Underweight"
        elif 18.5 <= bmi < 24.9:
            return "Normal weight"
        elif 25 <= bmi < 29.9:
            return "Overweight"
        else:
            return "Obesity"

#patient update pydantic model
class PatientUpdate(BaseModel):
    name: Annotated[Optional[str], Field(default=None)]
    city: Annotated[Optional[str], Field(default=None)]
    age: Annotated[Optional[int], Field(default=None, gt=0)]
    gender: Annotated[Optional[Literal['male', 'female','others']], Field(default=None)]
    height: Annotated[Optional[float], Field(default=None, gt=0)]
    weight: Annotated[Optional[float], Field(default=None, gt=0)]




"""helper functions to load and save data from/to JSON file
"""
# function to load data from JSON file
def load_data():
    with open('patients.json', 'r') as f:
        data = json.load(f)

    return data


# function to save data to JSON file
def save_data(data):
    with open("patients.json", "w") as file:
        json.dump(data, file, indent=4)


# define home route
@app.get("/")
async def hello():
    return {"message": "PATIENT MANAGEMENT SYSTEM API"}


# define about route
@app.get("/about")
async def about():
    return {"message": "This is a Patient Management System API built with FastAPI."}



# define view route to return all patient data
@app.get("/view")
async def view():
    data = load_data()
    return data



# define view route to return specific patient data by ID
@app.get("/view/{patient_id}")
async def view_patient(patient_id: str = Path(..., description="The ID of the patient to retrieve" , example="P001")):
    data = load_data()
    patient = data.get(patient_id)   # ✅ directly access dict by key
    if patient:
        return patient
    raise HTTPException(status_code=404, detail="Patient not found")



# define sort route to return patients sorted 
@app.get("/sort")
async def sort_patients(sort_by: str = Query(..., description="sort on the basis of height,weight or bmi"), order:str = Query("asc", description="Order of sorting: asc or desc")):
    
    valid_fields = {"height", "weight", "bmi"}
    if sort_by not in valid_fields:
        raise HTTPException(status_code=400, detail=f"Invalid sort field. Must be one of {valid_fields}")
    
    if order not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="Invalid order. Must be 'asc' or 'desc'")
    
    sort_order = True if order == "asc" else False
    
    data = load_data()
    sorted_data = sorted(data.items(), key=lambda item: item[1].get(sort_by, 0), reverse=(order==sort_order))


    return sorted_data


@app.post("/create")
async def create_patient(patient: Patient):
    # load existing data
    data = load_data()

    # check if patient with same ID already exists
    if patient.id in data:
        raise HTTPException(status_code=400, detail="Patient with this ID already exists")

    # add new patient to data
    data[patient.id] = patient.model_dump(exclude={"id"})


    # save updated data back to JSON file
    save_data(data)

    return JSONResponse(status_code=201, content={"message": "Patient created successfully"})


@app.put('/edit/{patient_id}')
async def  update_patient(patient_id : str , patient_update:PatientUpdate):
    
    #load data
    data = load_data()

    #check patient
    if patient_id not in data:
        raise HTTPException(status_code=404 , detail = "Patient Not Found")
    
    #extract data
    existing_patient_info = data[patient_id]

    #updated info
    updated_patient_info = patient_update.model_dump(exclude_unset=True)

    #update dictionary
    for key,value in updated_patient_info.items():
        existing_patient_info[key] = value

    #updated pydantic object
    existing_patient_info['id'] = patient_id
    patient_pydantic_obj = Patient(**existing_patient_info)


    #build dict
    existing_patient_info = patient_pydantic_obj.model_dump(exclude = {'id'})

    #add this dict to data
    data[patient_id] = existing_patient_info

    #save data
    save_data(data)

    return JSONResponse(status_code=200 , content={"message" : "patient_updated"})


@app.delete('/delete/{patient_id}')
def delete_patient(patient_id: str):

    # load data
    data = load_data()

    if patient_id not in data:
        raise HTTPException(status_code=404, detail='Patient not found')
    
    del data[patient_id]

    save_data(data)

    return JSONResponse(status_code=200, content={'message':'patient deleted'})


         

