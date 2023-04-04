# Author: Matt Williams
# Version: 4/3/2023


import os
import json
from utils import JSON_DIR_PATH

def save_to_json_file(data, file_name):

    if not os.path.exists(JSON_DIR_PATH):
        os.mkdir(JSON_DIR_PATH)

    file_path = os.path.join(JSON_DIR_PATH, file_name)

    if os.path.exists(file_path):
        os.remove(file_path)
    
    with open(file_path, mode = "w", encoding="utf-8") as json_file: 
        json.dump(data, json_file, indent = 1)


def load_from_json_file(file_name):
    file_path = os.path.join(JSON_DIR_PATH, file_name)

    if not os.path.exists(file_path):
        return None

    output = None
    with open(file_path, mode = "r", encoding = "utf-8") as json_file:
        output = json.load(json_file)


    return output 
