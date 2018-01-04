import json
from os import path as osp
import re

from config import *
from vocab import *
from tree import Tree

NodeTypeDict = dict()

def build(ps, table, NodeTypeDict):
    # input a "parse sentences"
    ss = re.sub(' +', ' ', ps.strip().replace('\n', ''))
    stack = []
    i = 0
    while i < len(ss):
        if ss[i] == '(':
            stack.append(Tree())
            i += 1
        elif ss[i] == ")":
            if len(stack) == 1:
                return stack[0]
            else:
                stack[-2].addChild(stack[-1])
                stack.pop()
            i += 1
        elif ss[i] == " ":
            i += 1
        else:
            j = i+1
            while (j < len(ss) and  ss[j] != "(" and ss[j] != ")" and ss[j] != " "):
                j += 1
            word = ss[i:j]
            if stack[-1].hasNodetype():
                stack[-1].word = word2id(table, word)
                stack[-1].raw_word = word
            else:
                stack[-1].node_type = word
                NodeTypeDict[word.upper()] = NodeTypeDict.get(word.upper(), 0) + 1
            i = j
    print("Error! If parsed sentence(ps) is right, function should return"
          " in while-loop not here!, sentence:\n %s" % ps)

def buildData(sp, vocab_table):
    file_a = osp.join(BASE_DIR, sp, 'a.txt.json')
    file_b = osp.join(BASE_DIR, sp, 'b.txt.json')
    file_sim = osp.join(BASE_DIR, sp, 'sim.txt')
    with open(file_sim, 'r') as fd:
        lines = fd.readlines()
        sim = [float(ix.strip()) for ix in lines]
    dataset = [] # (sa, sb, sim)
    with open(file_a, 'r') as fd, open(file_b, 'r') as fb:
        json_a = json.load(fd)
        json_b = json.load(fb)
    assert len(sim) == len(json_a['sentences'])
    assert len(sim) == len(json_b["sentences"])
    for i in range(len(sim)):
        tree_a = build(json_a['sentences'][i]['parse'], vocab_table, NodeTypeDict)  # sentence a
        tree_b = build(json_b['sentences'][i]['parse'], vocab_table, NodeTypeDict)
        tp = (tree_a, tree_b, sim[i])
        dataset.append(tp)
    return dataset, NodeTypeDict