import json
import sys

file_name=sys.argv[1]
print(len(json.load(open(file_name))))
