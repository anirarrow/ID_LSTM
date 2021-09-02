from pycorenlp import StanfordCoreNLP
from pyparsing import OneOrMore, nestedExpr
from nltk import Tree
import numpy as np
import nltk
import json
def tree2dict(tree):
	return {tree.label(): [tree2dict(t)  if isinstance(t, Tree) else t for t in tree]}

#def get_node_depth(t):
	#c=0
	'''
	n_leaves = len(t.leaves())
	leavepos = set(t.leaf_treeposition(n) for n in range(n_leaves))
	for pos in t.treepositions():
		if pos not in leavepos:
			sub_tree=t[pos]
			#print(sub_tree.leaves())
			if len(sub_tree.leaves())==1:
				print(sub_tree.label(),len(pos),sub_tree.leaves())
	'''
	#for sub_tree in t.subtrees(): 
    #if sub_tree.label()  == '<JJ>' and 'different' in set(sub_tree.leaves()):
    #print('yes')
	#	print(sub_tree.label())
	#	print(sub_tree.leaves())
#def recurse(d,child_dictionary):


nlp = StanfordCoreNLP('http://localhost:9000')
textInput='Musicland is trying to embrace the Internet and emulate the retail chains like Starbucks.'

fout=open('train_cgf.res','w')
output = nlp.annotate(textInput, properties={'annotators': 'parse','outputFormat': 'json','timeout': 1000})
#print(output['sentences'][0]["parse"])
pt=output['sentences'][0]['parse']
t = Tree.fromstring(pt)
t2d=tree2dict(t)
json.dump(t2d, fout)
print(type(t2d))
fout.close()
print("Convert bracketed string into tree:")
#print(type(t))
#print(t.__repr__())
#get_node_depth(t)
#exit(0)
#get_aaai_parsetree(pt)
#print('\n')
#print(pt)

'''
def get_aaai_parsetree(pt):
	stack=[]
	depth=-1
	d=dict()
	for i in range(0,len(pt)):
		ch=pt[i]
		if ch=='(':
			stack.append(ch)
			depth+=1
		elif ch!='(' and ch!=(')'):
			stack.append(ch)
		elif ch==')':
			char=stack.pop()
			string=char
			while char!='(':
				char=stack.pop()
				string+=char
			if len(stack)==0:
				break
			char=stack.pop()
			s=string[::-1]
			if len(s.split())==2:
				d[s.split()[1]]=depth
			depth-=1
	print(d)
'''