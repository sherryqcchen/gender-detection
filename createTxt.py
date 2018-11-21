import os
from bson.objectid import ObjectId
import DB as db
import sys
import re
import numpy as np

def choose_name(fullname):
	name_arr = re.split(r'\s|\-',fullname)
	for name in name_arr:
		if re.findall(r'\.|\/',name) or len(name)<2:
			continue
		else:
			return name
	return name_arr[-1]

female_files = [x for x in filter(lambda filename:filename.endswith('jpeg'),os.listdir(sys.argv[1]+'/female/'))]

female_ids =[x for x in map(lambda filename:filename.split('.')[0],female_files)]

male_files = [x for x in filter(lambda filename:filename.endswith('jpeg'),os.listdir(sys.argv[1]+'/male/'))]

male_ids = [x for x in map(lambda filename:filename.split('.')[0],male_files)]

re_co = db.get_researcher_copy()

male_names = []

female_names = []

missing_value = 0

for _id in male_ids:
	doc = list(re_co.find({'_id':ObjectId(_id)}))[0]
	name = doc['name']
	if name:
		firstname = choose_name(name).lower()
		male_names.append((firstname,_id))
	else:
		missing_value+=1

for _id in female_ids:
	doc = list(re_co.find({'_id':ObjectId(_id)}))[0]
	name = doc['name']
	if name:
		firstname = choose_name(name).lower()
		female_names.append((firstname,_id))
	else:
		missing_value+=1

with open(sys.argv[2],'w') as f:
	for name,_id in male_names:
		f.write(name+',M,'+_id+'\n')
	for name,_id in female_names:
		f.write(name+',F,'+_id+'\n')

print('successful created:%s'%(len(male_names)+len(female_names)))
print('missing name:%s'%(missing_value))




