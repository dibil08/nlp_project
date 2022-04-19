import sys, os, json
import matplotlib.pyplot as plt

class KarstInspector:

    semanticRelations = [
        "AFFECTS", 
        "HAS_ATTRIBUTE", 
        "HAS_CAUSE", 
        "CONTAINS", 
        "HAS_COMPOSITION", 
        "DEFINED_AS", 
        "HAS_FORM", 
        "HAS_FUNCTION", 
        "HAS_LOCATION", 
        "MEASURES", 
        "HAS_POSITION", 
        "HAS_RESULT", 
        "HAS_SIZE", 
        "STUDIES", 
        "OCCURS_IN_TIME"]

    definitionElements = [
        "DEFINIENDUM",
        "DEFINITOR",
        "GENUS", 
        "SPECIES"]

    def __init__(self, path):
        self.path = path  # path to annotated definitions
        self.count = {}
        self.index = {}

    def print(self):
        languages = list(self.count.keys())
        for language in languages:
            print(language)
            print(json.dumps(self.count[language]["semanticRelations"], indent=3))
            print(json.dumps(self.count[language]["definitionElements"], indent=3))

    def visualize(self):
        languages = list(self.count.keys())

        for language in languages:
            definitionElements = list(self.count[language]["definitionElements"].values())
            semanticRelations = list(self.count[language]["semanticRelations"].values())
            values = definitionElements + semanticRelations

            plt.figure()
            plt.title(language)
            plt.hist(
                self.definitionElements + self.semanticRelations, 
                len(self.definitionElements) + len(self.semanticRelations),
                weights = values)
            plt.xticks(rotation = 90)
            plt.tight_layout()

        plt.show()

    def inspect(self):
        for language in os.listdir(self.path):

            if not os.path.isdir(self.path + "/" + language):
                continue

            self.count[language] = {
                    "semanticRelations": self.getCountForm(self.semanticRelations),
                    "definitionElements": self.getCountForm(self.definitionElements)}

            self.index[language] = {
                    "semanticRelations": {},
                    "definitionElements": {}}

            self.exploreLanguage(language)

    def exploreLanguage(self, language):
        path = self.path + "/" + language

        for filename in os.listdir(path):
            if filename.endswith('.tsv'):
                self.exploreFile(language, filename)

    def exploreFile(self, language, filename):
        path = self.path + "/" + language + "/" + filename

        self.index[language][filename] = []

        file = open(path, "rb")

        for line in file:
            
            line = line.decode("utf-8") 
            
            if line[0] == "#":
                continue
            if line[0] in ["\n", "\r"]:
                continue

            line = line.replace("\\", "")[:-1]
            segments = line.split("\t")

            for segment in segments:
                for part in segment.split("|"):

                    for relation in self.semanticRelations:
                        if part.find(relation) != -1:
                            number = part[len(relation)+1:-1]
                            if number not in self.index:
                                self.count[language]["semanticRelations"][relation] += 1
                                self.index[language][filename].append(number)

                    for element in self.definitionElements:
                        if part.find(element) != -1:
                            number = part[len(relation)+1:-1]
                            if number not in self.index:
                                self.count[language]["definitionElements"][element] += 1
                                self.index[language][filename].append(number)

    def getCountForm(self, inputList):
        form = {}
        for relation in inputList:
            form[relation] = 0
        return form

if __name__ == "__main__":

    path = ""
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = "KarstAnnotatedDefinitions/"

    sri = KarstInspector(path)
    sri.inspect()
    sri.print()
    sri.visualize()
