import os
from fnmatch import fnmatch

class semRelDataInspector:

    def __init__(self, path) -> None:
        self.path = path
        self.relations = []

    def inspect(self):
        files = self.findTsvFiles()

        for file in files:
            file.replace("\\", "/")
            self.readFile(file)

        print(self.relations)

    def readFile(self, path):
        file = open(path, "rb")

        for line in file:
            
            line = line.decode("utf-8") 
            
            if len(line) < 2:
                continue
            if line[0] == "#" or line[1] == "#":
                continue
            if line[0] in ["\n", "\r"]:
                continue

            line = line.replace("\\", "")[:-1]
            segments = line.split("\t")

            if len(segments) >= 4:
                relations = segments[3]
                if relations != "_":
                    relations = relations.split("|")
                    for relation in relations:
                        if relation not in self.relations:
                            self.relations.append(relation)

    def findTsvFiles(self):
        tsvFiles = []
        for path, subdirs, files in os.walk(self.path):
            for name in files:
                tsvFiles.append(os.path.join(path, name))
        return tsvFiles



if __name__ == "__main__":

    # SemRelData
    path = "../datasets/semreldata-dataset/annotation"

    srdi = SemRelDataInspector(path)
    srdi.inspect()