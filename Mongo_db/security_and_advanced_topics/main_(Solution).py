from bson import ObjectId
from pymongo import MongoClient
from fastapi import FastAPI

### Setup MongoDB Client
mongodb_uri = "mongodb://localhost:27017/"
client = MongoClient(mongodb_uri)

### Setup FastAPI App
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/zips/{zip_id}")
async def get_zip_by_id(zip_id):
    return client.performance_db.zips.find_one({"_id": zip_id})

@app.get("/zips/city/{city}")
async def get_zip_by_city(city):
    return list(client.performance_db.zips.find({"city": city}))

@app.post("/users", status_code=201)
async def add_user(user_data: dict):
    insert_res = client.my_store.users.insert_one(user_data)
    return {"inserted_id": str(insert_res.inserted_id)}

@app.get("/users")
async def get_user(user_data: dict):
    user_doc = client.my_store.users.find_one(user_data)
    # user_doc["_id"] = user_doc["_id"].__repr__()
    user_doc["_id"] = str(user_doc["_id"])
    return user_doc 

@app.put("/users/{user_id}")
async def replace_user(user_id, user_data: dict):
    replace_res = client.my_store.users.replace_one({"_id": ObjectId(str(user_id))}, user_data)
    return replace_res.raw_result

@app.patch("/users/{user_id}")
async def update_user(user_id, user_data: dict):
    update_res = client.my_store.users.update_one({"_id": ObjectId(str(user_id))}, {"$set": user_data})
    return update_res.raw_result

@app.delete("/users/{user_id}")
async def delete_user(user_id):
    delete_res = client.my_store.users.delete_one({"_id": ObjectId(str(user_id))})
    return delete_res.raw_result