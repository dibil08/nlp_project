import os
from collections import Counter

class karstInspector:

    semantic_relations = [
        "AFFECTS", "HAS_ATTRIBUTE", "HAS_CAUSE", "CONTAINS", "HAS_COMPOSITION", "DEFINED_AS", 
        "HAS_FORM", "HAS_FUNCTION", "HAS_LOCATION", "MEASURES", "HAS_POSITION", "HAS_RESULT", 
        "HAS_SIZE", "STUDIES", "OCCURS_IN_TIME", "OCCURS_IN_MEDIUM", "COMPOSED_OF", "COMPOSITION_MEDIUM"]

    definition_elements = ["DEFINIENDUM", "DEFINITOR", "GENUS", "SPECIES"]

    languages = ["EN", "HR", "SL"]

    def __init__(self, path):
        self.path = path

        self.stats = self.getDictForm()

        self.getFiles()
        self.getSemanticRelations()
        self.getDefinitionElements()

    def getDictForm(self):
        stats = {}
        for language in self.languages:
            stats[language] = {
                "path": None,
                "file_count": 0,
                "filenames": [],
                "definition_elements": {},
                "semantic_relations": {}
            }
            for definition_element in self.definition_elements:
                stats[language]["definition_elements"][definition_element] = {
                    "count": 0,
                    "storage": []
                }
            for semantic_relation in self.semantic_relations:
                stats[language]["semantic_relations"][semantic_relation] = {
                    "count": 0,     
                    "storage": []
                }
        return stats

    def getFiles(self):
        for language in self.languages:
            path_language = self.path + "/" + language
            files =  os.listdir(path_language)
            self.stats[language]["path"] = path_language
            self.stats[language]["filenames"] = files
            self.stats[language]["file_count"] = len(files)

    def getElementIndex(self, element):
        element = element.split("[")
        if len(element) == 2:                   
            return element[1][:-1], element[0]
        else:
            return None, element[0]

    def getSemanticRelations(self):
        for language in self.languages:
            files = self.stats[language]["filenames"]      
  
            for file in files:
                file_path = self.stats[language]["path"] + "/" + file
                content = open(file_path, encoding="utf8")

                relation_builder = {}

                for line in content:
                    
                    if line[0] in ["\n", "\r", "#" ]:
                        continue

                    segments = line.split("\t")
                    word_index = segments[0]
                    char_index = segments[1] 
                    word = segments[2].lower()
                    category = segments[4]
                    definition_element = segments[5]
                    relations = segments[6]

                    if relations == "_":
                        continue

                    relations = segments[6].replace("\\", "").split("|")

                    for relation in relations:
                        index, relation_type = self.getElementIndex(relation)

                        if index == None:
                            self.stats[language]["semantic_relations"][relation]["count"] += 1
                            self.stats[language]["semantic_relations"][relation]["storage"].append(word)
                        else:
                            if relation in relation_builder.keys():
                                relation_builder[relation]["string"] += " " + word              
                            else:
                                relation_builder[relation] = {
                                    "string": word,
                                    "type": relation_type
                                }

                for values in relation_builder.values():
                    relation = values["type"]
                    self.stats[language]["semantic_relations"][relation]["count"] += 1
                    self.stats[language]["semantic_relations"][relation]["storage"].append(values["string"])


    def getDefinitionElements(self):
        for language in self.languages:
            files = self.stats[language]["filenames"]      
  
            for file in files:
                file_path = self.stats[language]["path"] + "/" + file
                content = open(file_path, encoding="utf8")

                definition_builder = {}

                for line in content:
                    
                    if line[0] in ["\n", "\r", "#" ]:
                        continue

                    segments = line.split("\t")
                    word_index = segments[0]
                    char_index = segments[1] 
                    word = segments[2].lower()
                    category = segments[4]
                    definitions = segments[5]
                    relations = segments[6]

                    if definitions == "_":
                        continue

                    definitions = definitions.split("|")

                    for definition in definitions:
                        index, definition_type = self.getElementIndex(definition)

                        if index == None:
                            self.stats[language]["definition_elements"][definition_type]["count"] += 1
                            self.stats[language]["definition_elements"][definition_type]["storage"].append(word)
                        else:
                            if definition in definition_builder.keys():
                                definition_builder[definition]["string"] += " " + word              
                            else:
                                definition_builder[definition] = {
                                    "string": word,
                                    "type": definition_type
                                }

                for values in definition_builder.values():
                    definition_type = values["type"]
                    if definition_type in self.definition_elements:
                        self.stats[language]["definition_elements"][definition_type]["count"] += 1
                        self.stats[language]["definition_elements"][definition_type]["storage"].append(values["string"])

    def printSemanticRelationsStats(self, language, relation=None):
        print("\nLanguage:", language)
        if relation:
            data = self.stats[language]["semantic_relations"][relation]
            print("{}: {}".format(data["count"], relation))
            item_counter = Counter(data["storage"]).most_common()
            for item in item_counter:
                print("\t\t{}: {}".format(item[1], item[0]))
        else:
            for key, value in self.stats[language]["semantic_relations"].items():
                print("{}: {}".format(key, value["count"]))
                item_counter = Counter(value["storage"]).most_common()
                for item in item_counter:
                    print("\t\t{}: {}".format(item[1], item[0]))

    def printDefinitionElementsStats(self, language, definition=None):
        print("\nLanguage:", language)
        if definition:
            data = self.stats[language]["definition_elements"][definition]
            print("{}: {}".format(data["count"], definition))
            item_counter = Counter(data["storage"]).most_common()
            for item in item_counter:
                print("\t\t{}: {}".format(item[1], item[0]))
        else:
            for key, value in self.stats[language]["definition_elements"].items():
                print("{}: {}".format(key, value["count"]))
                item_counter = Counter(value["storage"]).most_common()
                for item in item_counter:
                    print("\t\t{}: {}".format(item[1], item[0]))

    def getDefinitionElementsStats(self, language, definition=None):
        print("\nLanguage:", language)
        if definition:
            data = self.stats[language]["definition_elements"][definition]
            item_counter = Counter(data["storage"]).most_common()
            return item_counter
        else:
            counters=[]
            for key, value in self.stats[language]["definition_elements"].items():
                print("{}: {}".format(key, value["count"]))
                item_counter = Counter(value["storage"]).most_common()
                counters.append(item_counter)
            return counters



if __name__ == "__main__":

    path = "../datasets/karst/Karst Annotated Definitions/AnnotatedDefinitions"

    karst_stats = karstInspector(path)
    
    #karst_stats.printSemanticRelationsStats("EN")
    #karst_stats.printSemanticRelationsStats("EN", "HAS_CAUSE")

    #karst_stats.printDefinitionElementsStats("EN")
    karst_stats.printDefinitionElementsStats("EN", "DEFINITOR")
    #karst_stats.printDefinitionElementsStats("EN", "DEFINIENDUM")
    #karst_stats.printDefinitionElementsStats("EN", "SPECIES")
    #karst_stats.printDefinitionElementsStats("EN", "GENUS")
