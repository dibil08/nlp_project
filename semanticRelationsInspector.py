import sys, os, json
import matplotlib.pyplot as plt

class SemanticRelationInspector:

    semanticRelations = [
        "AFFECTS", "HAS_ATTRIBUTE", "HAS_CAUSE", "CONTAINS", "HAS_COMPOSITION", "DEFINED_AS", "HAS_FORM", 
        "HAS_FUNCTION", "HAS_LOCATION", "MEASURES", "HAS_POSITION", "HAS_RESULT", "HAS_SIZE", "STUDIES", "OCCURS_IN_TIME"]

    def __init__(self, dirPath):
        self.dirPath = dirPath # path to annotated definitions
        self.countRelations = {}
        self.indexRelations = {}

    def print(self):
        languages = list(self.countRelations.keys())
        for language in languages:
            print(language)
            print(json.dumps(self.countRelations[language], indent=3))

    def visualize(self):
        languages = list(self.countRelations.keys())
        for language in languages:
            plt.figure()
            plt.title(language)
            plt.hist(self.semanticRelations, len(self.semanticRelations), weights=tuple(self.countRelations[language].values()))
            plt.xticks(rotation = 90)
            plt.tight_layout()
        plt.show()

    def inspect(self):
        for element in os.listdir(self.dirPath):
            fullPath = self.dirPath + "/" + element
            if os.path.isdir(fullPath):
                self.countRelations[element] = self.countRelationsForm()
                self.exploreLanguage(fullPath)

    def exploreLanguage(self, langPath):
        language = langPath.split("/")[-1]
        for fileName in os.listdir(langPath):
            if fileName.endswith('.tsv'):
                self.exploreFile(language, langPath + "/" + fileName)

    def exploreFile(self, language, filePath):
        fileName = filePath.split("/")[-1]
        self.indexRelations[fileName] = []
        file = open(filePath, "rb")
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
                            indexNumber = part[len(relation)+1:-1]
                            if indexNumber not in self.indexRelations:
                                self.countRelations[language][relation] += 1
                                self.indexRelations[fileName].append(indexNumber)

    def countRelationsForm(self):
        return {
            "AFFECTS": 0, "HAS_ATTRIBUTE": 0, "HAS_CAUSE": 0, "CONTAINS": 0, "HAS_COMPOSITION": 0, "DEFINED_AS": 0, "HAS_FORM": 0,
            "HAS_FUNCTION": 0, "HAS_LOCATION": 0, "MEASURES": 0,"HAS_POSITION": 0,"HAS_RESULT": 0,"HAS_SIZE": 0,"STUDIES": 0, "OCCURS_IN_TIME": 0}

if __name__ == "__main__":

    path = ""
    if len(sys.argv) == 2:
        path = sys.argv[1]
    else:
        path = "karst/karst_annotated_definitions/Karst Annotated Definitions/AnnotatedDefinitions/"

    sri = SemanticRelationInspector(path)
    sri.inspect()
    sri.print()
    sri.visualize()
