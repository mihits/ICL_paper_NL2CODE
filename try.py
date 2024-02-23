import json

with open('saved_demos.json', 'r') as f:
    data = json.load(f)


print(type(data))
