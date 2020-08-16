import json

class Constant(float):
    def __new__(cls, value):
        self = float.__new__(cls, value["value"])  # KeyError if missing "value"
        self.units = value.get("units", None)
        self.doc = value.get("doc", None)
        return self

class Constants():
    # load the json file into a dictionary of Constant objects
    def __init__(self):
        with open("./data/constants.json") as fh:
            json_object = json.load(fh)

        # create a new dictionary
        self.constants_dict = {}
        for constant in json_object.keys():
            # put each Constant into it
            self.constants_dict[constant] = Constant(json_object[constant])

    # try to get the requested attribute
    def __getattr__(self, name):
        # missing keys are returned None, use self.constants_dict[name]
        # if you want to raise a KeyError instead
        return self.constants_dict.get(name, None)
