from KarstInspector import karstInspector


path = "./datasets/karst/Karst Annotated Definitions/AnnotatedDefinitions"

karst_stats = karstInspector(path)

stats=karst_stats.getDefinitionElementsStats("EN", "DEFINITOR")
print(stats)
stats=[(' are ', 80), (' is a ', 53), (' is ', 18), (' is defined as ', 11), (' the term ', 11), (' are known as ', 7), (' is the ', 7), (' are defined as ', 7), (' refers to ', 6), (' known as ', 6), (' is an ', 4), (' are an ', 4), (' are called ', 3), (' are a ', 3), (' to describe ', 3), (' also known as ', 2), (' is a type of ', 2), (' is a term describing ', 2), (' is defined as a ', 2), (' refers to a ', 2), (' is used to describe ', 2), (' as a ', 2), (' is called ', 1), (' is strictly applied to a ', 1), (' is called a ', 1), (' could be designated by the term ', 1), (' is called the ', 1), (' is termed ', 1), (' is now used fairly widely for a ', 1), (' are a particular type of ', 1), (' define ', 1), (' refers to the ', 1), (' is a related term ', 1), (' as that ', 1), (' signifies ', 1), (' are best defined as ', 1), (' is sometimes used to refer ', 1), (' is a term used ', 1), (' are typical of ', 1), (' is commonly used ', 1), (' a spectacular form of ', 1),(' is the term ', 1), (' is a specific category of ', 1), (' is a special type of ', 1), (' is known as ', 1), (' they are known as ', 1), (' are the reverse of ', 1), (' is sometimes applied to ', 1), (' a typical ', 1), (' a common term used for ', 1), (' represents ', 1), (' describes ', 1), (' is a general term referring to ', 1), (' are ', 1), (' is a common term used for ', 1), (' may be defined as ', 1), (' applies to ', 1), (' is the term applied to ', 1), (' means ', 1), (' designates a ', 1), (' are a type of ', 1), (' it has been used since a long time to indicate ', 1), (' is the term used to describe a ', 1), (' is a common group term ', 1), (' defined it as a ', 1), (' can be defined as ', 1), (' is used for ', 1), (' clay ', 1), (' term ', 1), (' includes ', 1), (' known collectively as ', 1), (' referred to as ', 1), (' a particular type of ', 1), (' defined ', 1), (' meaning a ', 1), (' is a generic term that refers to ', 1), (' is frequently used as a ', 1), (' has been used for ', 1), (' that is , ', 1), (' is a subtype of ', 1), (' is part of a ', 1), (' is a group of ', 1), (' also termed ', 1)]
i=0
curr=0
with open('./karst_bert/filtered_english_corpus.txt', 'w', encoding="utf8") as wf:
    with open("./karst_bert/english_corpus__project_fri.txt", "r", encoding="utf8") as f:
        for line in f:
            curr+=1
            if len(line)>10:
                for stat in stats:
                    if stat[0] in line:
                      #  print(curr)
                        wf.write(line)
                        i+=1
                        break
print(f"got {i} lines")
