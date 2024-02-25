import json

n = "ho"
j = "ko000"

output_file_path = r"xx" + n + "_" + j.split('/')[-1] + ".json"
with open(output_file_path, 'w') as json_file:
    json.dump({"h":"g"}, json_file, indent=2)
