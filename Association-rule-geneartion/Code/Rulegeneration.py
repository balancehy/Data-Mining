#!/usr/bin/env python3
import numpy as np
import os
import pickle
import sys
import copy


class RuleGen():
    def __init__(self, support_dict=None):
        self._data = None
        self._minsup = 0.5
        self._minconf = 0.7
        self.diseases = ["ALL", "Breast Cancer", "Colon Cancer", "AML"]
        if support_dict is not None:
            self._support_dict = copy.deepcopy(support_dict)
            print("Imported support count dictionary")
        else:
            self._support_dict = {}
            print("Will generate support count dictionary")

    def read_data(self, fpath):
        with open(fpath, "r") as f:
            lines = f.readlines()
        res = []
        for line in lines:
            res.append(line.rstrip().split('\t'))
        return res

    def data2numpy(self, data):
        r = len(data)
        c = len(data[0]) - 1  # Exclude desease name
        res = np.zeros([r, c], dtype=np.int8)
        for i, line in enumerate(data):
            for j in range(len(line) - 1):
                if line[j] == "Up":
                    res[i, j] = 1
        self._data = res

    def gen_supp_dict(self, filename):
        data = np.genfromtxt(filename, delimiter='\t', dtype=str)
        total = len(data)
        key_map = {}
        idx = 0
        print("total patient is {}".format(total))
        data_t = data.transpose()
        freq_dict = {}
        freq_by_depth = [[]]
        depth = 0
        for i in range(total):

            down = set(np.where(data_t[i] == "Down")[0])
            up = set(np.where(data_t[i] == "Up")[0])
            if len(down) / total >= self._minsup:
                down_key = str(i+1) + "0"
                freq_dict[down_key] = down
                key_map[down_key] = idx
                idx += 1
                freq_by_depth[depth].append((down_key, down_key))

            if len(up) / total >= self._minsup:
                up_key = str(i+1) + "1"
                freq_dict[up_key] = up
                key_map[up_key] = idx
                idx += 1
                freq_by_depth[depth].append((up_key, up_key))
        for i, key in enumerate(self.diseases):
            disease_idx = set(np.where(data_t[100] == key)[0])
            if len(disease_idx) / total >= self._minsup:
                disease_key = str(101) + str(i)
                freq_dict[disease_key] = disease_idx
                key_map[disease_key] = idx
                idx += 1
                freq_by_depth[depth].append((disease_key, disease_key))
                print(disease_key)
        freq_by_depth[depth]

        freq_by_depth.append([])

        while len(freq_by_depth[depth]) != 0:

            for i in range(len(freq_by_depth[depth])):
                key1, last1 = freq_by_depth[depth][i]
                for j in range(key_map[last1] + 1, len(freq_by_depth[0])):
                    key2, last2 = freq_by_depth[0][j]
                    left_set = freq_dict[key1]
                    right_set = freq_dict[key2]
                    intersect = left_set.intersection(right_set)
                    if len(intersect) / total >= self._minsup:
                        freq_key, last = self.merge_key(key1, key2)
                        if freq_key not in freq_dict:
                            freq_dict[freq_key] = intersect
                            freq_by_depth[depth + 1].append((freq_key, last))

            depth += 1
            if len(freq_by_depth[depth]) != 0:
                freq_by_depth.append([])

        self._support_dict = freq_dict

        out_file = "freq_count_supp{}.txt".format(self._minsup)
        total_freq = 0
        with open(out_file, "w") as f:
            f.write("Support is set to be: {}%\n".format(self._minsup * 100))
            for i in range(depth):
                f.write("number of length-{} lengths frequent itemsets: {}\n".format(i + 1, len(freq_by_depth[i])))
                total_freq += len(freq_by_depth[i])
            f.write("number of all lengths frequent itemsets: {}\n".format(total_freq))
            f.close()
        return freq_dict

    def merge_key(self, key1, key2):
        set1 = set(key1.split(","))
        set2 = set(key2.split(","))
        new_set = list(set1.union(set2))
        last_elem = new_set[-1]
        new_set.sort(key=lambda x: int(x))
        # print(new_set)
        new_key = ",".join(new_set)
        return new_key, last_elem

    def encode(self, indexlist, freqlist):
        key = ""
        for i, x in enumerate(indexlist):  # encode key
            if i != len(indexlist) - 1:
                key += freqlist[x] + ","
            else:
                key += freqlist[x]
        return key

    def decode(self, key):
        num = key[:-1]
        if key[-1]=="1":
            dirs = "Up"
        else:
            dirs = "Down"
        return "G"+str(num)+"_"+dirs
    
    def decode_itemset(self, keylist):
        string = "{"
        for i, k in enumerate(keylist):
            if i!=len(keylist)-1:
                string += self.decode(k) + ","
            else:
                string += self.decode(k) + "}"
        return string

    def count_support(self, indexset, freqlist):
        indexlist = list(indexset)
        key = self.encode(indexlist, freqlist)
        if key in self._support_dict:
            return len(self._support_dict[key])
        # Calculate support count
        logbit = None
        for x in indexlist:
            feat_idx = int(freqlist[x][:-1])  # gene number
            feat_val = int(freqlist[x][-1])  # gene up or down
            if logbit is None:
                logbit = self._data[:, feat_idx] == feat_val
            else:
                logbit = logbit & (self._data[:, feat_idx] == feat_val)

        # self._support_dict[key] = np.where(logbit)[0]
        return len(np.where(logbit)[0])

    def generate_rule(self, keystring):
        onefreqlist = keystring.split(",")  # decode from string to list
        n = len(onefreqlist)
        res_left = []
        res_right = []
        if n == 1:
            return res_left, res_right
        total_set = set([x for x in range(n)])  # search by index, total,left,right are all index set
        left_sets = []
        right_sets = []
        for i in range(n):
            left = total_set - {i}
            if len(self._support_dict[keystring]) / self.count_support(left, onefreqlist) > self._minconf:
                left_sets.append(left)
                right_sets.append({i})
                res_left.append([onefreqlist[x] for x in list(left)])
                res_right.append([onefreqlist[i]])

        helpset = set()
        while len(left_sets) > 0 and len(left_sets[0]) > 1:
            newleft_sets = []
            newright_sets = []
            
            for i in range(len(left_sets) - 1):
                for j in range(i + 1, len(left_sets)):
                    newleft = left_sets[i].intersection(left_sets[j])
                    if len(self._support_dict[keystring]) / self.count_support(newleft, onefreqlist) > self._minconf:
                        newright = right_sets[i].union(right_sets[j])
                        key = self.encode(list(newleft), onefreqlist) + ":" + \
                            self.encode(list(newright), onefreqlist)

                        if key not in helpset:
                            helpset.add(key)
                            newleft_sets.append(newleft)
                            newright_sets.append(newright)
                            res_left.append([onefreqlist[x] for x in newleft])
                            res_right.append([onefreqlist[x] for x in newright])

            left_sets = newleft_sets
            right_sets = newright_sets
        # print(helpset)
        return res_left, res_right

    def generate_all_rule(self):
        # self._rule_dict = {}
        keys = self._support_dict.keys()
        # print(keys)
        self._head = []
        self._body = []
        for key in keys:
            left, right = self.generate_rule(key)
            self._head += left
            self._body += right

    def template1(self, title, number, genelist):
        keylist = []
        for g in genelist:
            gnum, gdir = g.split("_")
            num = int(gnum[1:])
            if gdir == "Up":
                keylist.append(str(num) + "1")
            elif gdir == "Down":
                keylist.append(str(num) + "0")
        res = []
        count = 0
        for i in range(len(self._head)):
            head = set(self._head[i])
            body = set(self._body[i])
            num = 0
            if title == "RULE":
                for key in keylist:
                    if (key in head) or (key in body):
                        num += 1
            if title == "HEAD":
                for key in keylist:
                    if key in head:
                        num += 1
            if title == "BODY":
                for key in keylist:
                    if key in body:
                        num += 1
            if (number == "ANY" and num >= 1) or (number == "NONE" and num == 0) or (number == "1" and num == 1):
                res.append(self.decode_itemset(self._head[i])+ "->"+ self.decode_itemset(self._body[i]))
                count += 1
        return res, count

    def template2(self, title, length):
        res = []
        count = 0
        for i in range(len(self._head)):
            if (title == "HEAD" and len(self._head[i])>=length) or \
                (title == "RULE" and len(self._head[i])+len(self._body[i])>=length) or \
                (title == "BODY" and len(self._body[i])>=length):

                res.append(self.decode_itemset(self._head[i])+ "->"+ self.decode_itemset(self._body[i]))
                count += 1
        return res, count

    def template3(self, oper, *args):
        if oper == "1or1" or oper == "1and1":
            t1, n1, genelist1, t2, n2, genelist2 = args
            res1, _ = self.template1(t1, n1, genelist1)
            res2, _ = self.template1(t2, n2, genelist2)
        elif oper == "2or2" or oper == "2and2":
            t1, n1, t2, n2 = args
            res1, _ = self.template2(t1, n1)
            res2, _ = self.template2(t2, n2)
        elif oper == "1or2" or oper == "1and2":
            t1, n1, genelist1, t2, n2 = args
            res1, _ = self.template1(t1, n1, genelist1)
            res2, _ = self.template2(t2, n2)
        elif oper == "2or1" or oper == "2and1":
            t1, n1, t2, n2, genelist2 = args
            res1, _ = self.template2(t1, n1)
            res2, _ = self.template1(t2, n2, genelist2)

        s1 = set(res1)
        s2 = set(res2)
        # res = []
        if oper == "1or1" or oper == "1or2" or oper == "2or1" or oper == "2or2":
            res = list(s1.union(s2))
        elif oper == "1and1" or oper == "1and2" or oper == "2and1" or oper == "2and2":
            res = list(s1.intersection(s2))
        
        return res, len(res)

    def ofile(self, listofstring, count, fpath):
        listofstring.sort()
        with open(fpath, "w") as f:
            f.write(str(count)+"\n")
            for o in listofstring:
                f.write(o+"\n")
        

if __name__ == "__main__":
    fpath = sys.argv[1]
    asso_rule = RuleGen()
    data = asso_rule.read_data(fpath)
    # min_supp = 0.7
    # if len(sys.argv) == 3:
    #     min_supp = float(sys.argv[2])

    asso_rule.data2numpy(data)
    supp_dict = asso_rule.gen_supp_dict(fpath)
    # print(supp_dict["591,721,960"])
    # print(len(supp_dict["591,721"]))
    # print(test.count_support([0,1],["591","721","960"]))
    asso_rule.generate_all_rule()
    # print([len(test._head[i])+len(test._body[i]) for i in range(len(test._head))])
    # print(test.generate_rule("591,721,960"))

    # print(test.template1("RULE", "ANY", ["G59_Up"]))
    # print(test.template1("HEAD", "ANY", ["G10_Down"]))
    # print(test.template3("1or1", "HEAD", "ANY", ["G10_Down"], "BODY", 1, ["G59_Down"]))
    
    # a,_ = test.template2("HEAD", 1)
    # b, _= test.template2("BODY", 1)
    # print(set(a).intersection(set(b)))
    
    # res, count = test.template3("1and2", "HEAD", "ANY", ["G10_Down"], "BODY", 2)
    # res, count = test.template1("HEAD", "ANY", ["G10_Down"])
    res, count = asso_rule.template1("RULE", "ANY", ["G59_Up"])
    asso_rule.ofile(res, count, "./result11.txt")
    res, count = asso_rule.template1("RULE", "NONE", ["G59_Up"])
    asso_rule.ofile(res, count, "./result12.txt")
    res, count = asso_rule.template1("RULE", "1", ["G59_Up", "G10_Down"])
    asso_rule.ofile(res, count, "./result13.txt")
    res, count = asso_rule.template1("HEAD", "ANY", ["G59_Up"])
    asso_rule.ofile(res, count, "./result14.txt")
    res, count = asso_rule.template1("HEAD", "NONE", ["G59_Up"])
    asso_rule.ofile(res, count, "./result15.txt")
    res, count = asso_rule.template1("HEAD", "1", ["G59_Up", "G10_Down"])
    asso_rule.ofile(res, count, "./result16.txt")
    res, count = asso_rule.template1("BODY", "ANY", ["G59_Up"])
    asso_rule.ofile(res, count, "./result17.txt")
    res, count = asso_rule.template1("BODY", "NONE", ["G59_Up"])
    asso_rule.ofile(res, count, "./result18.txt")
    res, count = asso_rule.template1("BODY", "1", ["G59_Up", "G10_Down"])
    asso_rule.ofile(res, count, "./result19.txt")
    
    res, count = asso_rule.template2("RULE", 3)
    asso_rule.ofile(res, count, "./result21.txt")
    res, count = asso_rule.template2("HEAD", 2)
    asso_rule.ofile(res, count, "./result22.txt")
    res, count = asso_rule.template2("BODY", 1)
    asso_rule.ofile(res, count, "./result23.txt")

    res, count = asso_rule.template3("1or1", "HEAD", "ANY", ["G10_Down"], "BODY", 1, ["G59_Up"])
    asso_rule.ofile(res, count, "./result31.txt")
    res, count = asso_rule.template3("1and1", "HEAD", "ANY", ["G10_Down"], "BODY", 1, ["G59_Up"])
    asso_rule.ofile(res, count, "./result32.txt")
    res, count = asso_rule.template3("1or2", "HEAD", "ANY", ["G10_Down"], "BODY", 2)
    asso_rule.ofile(res, count, "./result33.txt")
    res, count = asso_rule.template3("1and2", "HEAD", "ANY", ["G10_Down"], "BODY", 2)
    asso_rule.ofile(res, count, "./result34.txt")
    res, count = asso_rule.template3("2or2", "HEAD", 1, "BODY", 2)
    asso_rule.ofile(res, count, "./result35.txt")
    res, count = asso_rule.template3("2and2", "HEAD", 1, "BODY", 2)
    asso_rule.ofile(res, count, "./result36.txt")
