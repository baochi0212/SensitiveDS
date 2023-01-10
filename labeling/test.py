import json

path = "/home/xps/educate/code/hust/DS_20222/data-science-e10/source/active-learning/data/sensitive_test.json"
files = []
#gotta read to save data before write new files !!!!
data = json.load(open(path, 'r'))
f_write = open(path, 'w')
for file in data:
    if file not in files:
        files.append(file)
    else:
        print(file)

json.dump(files, f_write, indent=3)
f_write.close()
#recheck:
print(f"DOUBLE CHECK: {len(json.load(open(path, 'r')))}")