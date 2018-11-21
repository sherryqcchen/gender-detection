from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27018")	

db = client['GoogleScholar']

def get_researcher_copy():
	return db['researcher__copies']

def get_researcher():
	return db['researchers']




