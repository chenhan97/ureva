import os

path_list = ["train.txt"]
#path_list = ["dev.txt.filtered"]
ent_list = []
ent_dict = {}
with open("dict.entity", encoding="utf-8", mode="w") as ent_writer, open("dict.ent_wf", encoding="utf-8", mode="w") as freq_writer:
    for path in path_list:
        with open(path, encoding="iso-8859-1", mode="r") as reader:
            for index,line in enumerate(reader):
                ent1, ent2 = line.split("\t")[1], line.split("\t")[2]
                if ent1 not in ent_list:
                    ent_writer.write(ent1+"\n")
                if ent2 not in ent_list:
                    ent_writer.write(ent2+"\n")
                if ent1 not in  ent_dict.keys():
                    ent_dict[ent1] = 1
                else:
                    ent_dict[ent1] += 1  
                if ent2 not in  ent_dict.keys():
                    ent_dict[ent2] = 1
                else:
                    ent_dict[ent2] += 1 
    for ent in ent_dict.keys():
        freq_writer.write(ent+"\t"+str(ent_dict[ent])+"\n")
                
