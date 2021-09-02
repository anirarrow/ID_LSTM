from pycorenlp import StanfordCoreNLP
from pyparsing import OneOrMore, nestedExpr
from nltk import Tree
import numpy as np
import nltk




def tree2dict(tree):
	return {tree.label(): [tree2dict(t)  if isinstance(t, Tree) else t for t in tree]}
def recurse(child_dictionary,depth,d):
	#print('***')
	#print(child_dictionary)
	print("OK")
	if type(child_dictionary)==list:
		child_dictionary=child_dictionary[0]
	print(child_dictionary)
	k=child_dictionary.keys()[0]
	depth+=1
	d['depth']=depth
	if type(child_dictionary[k][0])!=dict:
		print("###")
		return(child_dictionary[k])
	d['children']=recurse(child_dictionary[k],depth,d)
	#print("HERE:")
	#print(d)
	return(d)
nlp = StanfordCoreNLP('http://localhost:9000')
textInput='Musicland is trying to embrace the Internet and emulate the retail chains like Starbucks.'


output = nlp.annotate(textInput, properties={'annotators': 'parse','outputFormat': 'json','timeout': 1000})
#print(output['sentences'][0]["parse"])
pt=output['sentences'][0]['parse']
t = Tree.fromstring(pt)
t2d=tree2dict(t)
depth=0
#for key in t2d.keys():
#	if key in ['NP','NNP','S','VP','VBZ','TO','VB','DT','NN','CC','JJ','PP','IN','.']:
#		depth+=1:
d=dict()
t2d={'S': [{'NP': [{'NNP': ['Musicland']}]}]}
#for key in t2d.keys():
print(t2d.keys())
#final_dict=recurse(t2d[t2d.keys()[0]][0],depth,d)
final_dict=recurse(list(t2d.keys())[0],depth,d)
print(final_dict)
#for item in final_dict.keys():
#	print(final_dict[item])
