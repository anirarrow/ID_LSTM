import spacy
from nltk import Tree
import json
import sys
from collections import defaultdict
import pandas as pd
import json
import numpy as np


def tree2dict(tree):
	print("###")
	print("tree="+str(tree))
	#try:
	return {tree.label(): [tree2dict(t)  if isinstance(t, Tree) else t for t in tree]}
	#except:
	#	return None
    #return {tree.node: [tree2dict(t)  if isinstance(t, Tree) else t.label() for t in tree.treepositions()]}


#en_nlp = spacy.load('en')
#sentence="The quick brown fox jumps over the lazy dog."
#sentence="Musicland is trying to embrace the Internet and emulate the atmosphere of retail chains like Starbucks."

#print(doc)
def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_


def convert_to_input(d,dict_of_depths):
    #maindict=dict()
    if isinstance(d,dict):
        key=list(d.keys())[0]
        keydepth=dict_of_depths[key][0]
        return([{"depth":keydepth+2,"word":key},{"depth":keydepth+2,"children":[convert_to_input(d[key][i],dict_of_depths)for i in range(0,len(d[key]))][0]}])
        #for i in range(0,len(d[key])):
        	#value=convert_to_input(d[key][i],dict_of_depths)
        	#return([{"depth":keydepth+2,"word":key},{"depth":keydepth+2,"children":value}])
    else:
        loi=[]
        #print("$$$")
        #print(d)
        #for item in d:
        key=d
        #print(key)
        keydepth=dict_of_depths[d][0]
        loi.append({"depth":keydepth+2,"word":key})
        return(loi)

#[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
#print('\n')
def convert(o):
	if isinstance(o, np.generic): return o.item()
	raise TypeError


if __name__ == '__main__':
	en_nlp=spacy.load("en_core_web_sm")
	df=pd.read_csv("../idfc_data.csv")
	#print(df.head())
	fout=open("idfc_parse_trees.csv","w")
	for i in range(0,df.shape[0]):
		sentence = df.iloc[i,1]
		label = df.iloc[i,3]

		#exit(0)
		if isinstance(sentence,str):
			doc = en_nlp(sentence)
			for sent in doc.sents:
				#print("SENT:"+str(sent))
				if len(sent)>2:
					dict_of_depths=defaultdict(list)
					tree=to_nltk_tree(sent.root)
					print(tree)
					print('\n\n')
					d=tree2dict(tree)
					#print(d)
					#print("*************************")
					n_leaves = len(tree.leaves())
					leavepos = set(tree.leaf_treeposition(n) for n in range(n_leaves))
					for pos in tree.treepositions():
						if pos not in leavepos:
						    dict_of_depths[tree[pos].label()].append(len(pos))
						else:
						    dict_of_depths[tree[pos]].append(len(pos))
					print("%%%")
					pt = convert_to_input(d,dict_of_depths)

					final_pt=dict()
					final_pt["rating"]= label
					final_pt["depth"]=1
					final_pt["children"]=pt
					json.dump(final_pt,fout,default=convert)
					fout.write('\n')
					#for item in pt:
					#	print(item)
					#fout.write(str(final_pt)+'\n')
	fout.close()
