import os
import pathlib
import collections
import json
import string
import pandas
from configurations import Configuration

class Preprocess:
    def __init__(self) -> None:
        self.config = Configuration()

    def csv_r(self, path):
        return list(pandas.read_csv(path)["code"])
    
    def json_r(path):
        with open(path, "r") as file:
            loaded = json.load(file)
        return loaded
    
    def clang_w(self, location, idx, data):
        with open(os.path.join(location, f"{idx}.c"), "w") as cfile:
            cfile.write(data)
    
    def c_gen(self, location, data, num = 1):
        for idx, file in enumerate(data):
            self.clang_w(location, idx + num, file)

    def get_asts(ast_dir):
        out = {}
        for root, _, files in os.walk(ast_dir):
            for file in files:
                if file.endswith(".txt"):
                    source_file_pth = os.path.join(root, file) 
                    source_file_name = int(pathlib.Path(source_file_pth).stem.split(".")[0])
                    try:
                        with open(source_file_pth, "r") as source_file:
                            out[source_file_name] = source_file.read()
                    except OSError as e:
                        raise e
        return list(collections.OrderedDict(sorted(out.items())).values())
    
    def extract_features(self, observations, features):
        nodes = []
        for observation in observations:
            temp = []
            for data in observation.split():
                if data.endswith(features):
                    temp.append(data)
            nodes.append(temp)
        out = []
        for node in nodes:
            temp = []
            for feature in node:
                temp.append(feature.translate(str.maketrans("", "", string.punctuation)))
            out.append(temp)
        return out

if __name__ == "__main__":
    Preprocess()