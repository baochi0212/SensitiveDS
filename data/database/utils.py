from pymongo import MongoClient

client_tie = "mongodb+srv://baochi0212:Baochi2002@cluster0.1wvuto3.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(client_tie)
print(client)
db = client.db_test
for i in db['testcollection'].find():
    print(i)
