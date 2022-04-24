import sys, os, json
import matplotlib.pyplot as plt

class WnInspector:

    elements = [
        "exe",
        "def",
        "lemma"]

    def __init__(self, path):
        self.path = path
        self.count = {
            "lemmas": 0,
            "examples": 0,
            "synonyms": 0,
            "definitions": 0,
            "POS": {
                "noun": 0,
                "verb": 0,
                "adjective": 0,
                "adverb": 0}}

        self.index = {
                "lemma": [],
                "def": [],
                "exe": []}

    def print(self):
        print(json.dumps(self.count, indent=4))

    def visualize(self):

        plt.figure()
        plt.hist(
            ["lemmas", "examples", "synonyms", "definitions"],
            4,
            weights = list(self.count.values())[0:4])
        plt.xticks(rotation = 90)
        plt.tight_layout()

        plt.show()

    def inspect(self):

        pos_map = {
            "a": "adjective",
            "r": "adverb",
            "n": "noun",
            "v": "verb"}

        file = open(path, "rb")

        for line in file:
            
            line = line.decode("utf-8") 

            if line[0] == "#":
                continue

            segments = line.split("\t")
            offset = segments[0].split("-")[0]
            posTag = pos_map[segments[0].split("-")[1]]
            type = segments[1].split(":")[1]

            if offset not in self.index[type]:
                self.index[type].append(offset)
                if type == "exe":
                    self.count["examples"] += 1
                elif type == "def":
                    self.count["definitions"] += 1
                elif type == "lemma":
                    self.count["lemmas"] += 1

            if type == "lemma":
                self.count["synonyms"] += 1

            self.count["POS"][posTag] += 1

        
if __name__ == "__main__":

    path = ""
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = "wnSlv/wn-data-slv.tab"

    sri = WnInspector(path)
    sri.inspect()
    sri.print()
    sri.visualize()
