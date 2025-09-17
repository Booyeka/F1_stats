import os
import json

# folder containing your animations
folder_path = "graphing/animations/finsihed_animations"   # change to your folder
json_file = "graphing/animations/animations.json"


data = []


# iterate over files in folder
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
    
        if filename[1] == "_":
            k = filename[0]
            i = filename[2:]
        else:
            k = filename[:2]
            i = filename[3:]
        race_d = {
            "race" : k,
            "name" : i,
            "file" : "f1_animations/"+filename
        }


        data.append(race_d)
data.sort(key=lambda x:int(x["race"]))



# print(race_dict)
    
# print(race_dict)

# write back to JSON
with open(json_file, "w") as f:
    json.dump(data, f, indent=4)

