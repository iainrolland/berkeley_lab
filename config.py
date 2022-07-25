import json


def get_username():
    with open("config.json", "r") as f:
        details = json.load(f)
    return details["username"]


def get_password():
    with open("config.json", "r") as f:
        details = json.load(f)
    return details["password"]
