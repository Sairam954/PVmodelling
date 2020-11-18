import json
import pprint

def readJsonConfig(filepath):
    config_file = open(filepath,'r')
    config_dict = json.load(config_file)
    return config_dict
