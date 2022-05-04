import json
import numpy as np
import os

hard = 0.5

result = []

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

data = []
with open("data/c3_release/c3-d-train.json", "r", encoding='utf8') as f:
    data += json.load(f)
with open("data/c3_release/c3-m-train.json", "r", encoding='utf8') as f:
    data += json.load(f)

split0 = len(data)

with open("data/cn/gb/cat_gb_1.json", "r", encoding='utf8') as f:
    data += json.load(f)
with open("data/cn/gb/cat_gb_2.json", "r", encoding='utf8') as f:
    data += json.load(f)

split1 = len(data)

with open("data/cn/lb/cat_lb_1.json", "r", encoding='utf8') as f:
    data += json.load(f)
with open("data/cn/lb/cat_lb_2.json", "r", encoding='utf8') as f:
    data += json.load(f)

split2 = len(data)

with open("data/cn/ib/cat_ib_1.json", "r", encoding='utf8') as f:
    data += json.load(f)
with open("data/cn/ib/cat_ib_2.json", "r", encoding='utf8') as f:
    data += json.load(f)

split3 = len(data)

with open("data/cn/ct/cat_ct_1.json", "r", encoding='utf8') as f:
    data += json.load(f)
with open("data/cn/ct/cat_ct_2.json", "r", encoding='utf8') as f:
    data += json.load(f)

print(split0, split1, split2, split3, len(data))

nchoice = []
for i in range(len(data)):
    for j in range(len(data[i][1])):
        nchoice += [len(data[i][1][j]["choice"])]

split0n = 0
split1n = 0
split2n = 0
split3n = 0
for i in range(split0):
    split0n += len(data[i][1])
for i in range(split1):
    split1n += len(data[i][1])
for i in range(split2):
    split2n += len(data[i][1])
for i in range(split3):
    split3n += len(data[i][1])
    
print(split0n, split1n, split2n, split3n, len(nchoice))

weights = []
w = []
for i in range(len(nchoice)):
    if i < split0n:
        w += [(1-hard)/4]
    elif i < split1n:
        w += [1-hard]
    else:
        w += [0]
weights += [w]
w = []
for i in range(len(nchoice)):
    if i < split0n:
        w += [(1-hard)/4]
    elif i < split1n:
        w += [0]
    elif i < split2n:
        w += [1-hard]
    else:
        w += [0]
weights += [w]
w = []
for i in range(len(nchoice)):
    if i < split0n:
        w += [(1-hard)/4]
    elif i < split2n:
        w += [0]
    elif i < split3n:
        w += [1-hard]
    else:
        w += [0]
weights += [w]
w = []
for i in range(len(nchoice)):
    if i < split0n:
        w += [(1-hard)/4]
    elif i < split3n:
        w += [0]
    else:
        w += [1-hard]
weights += [w]
w = []
for i in range(len(nchoice)):
    w += [hard]
weights += [w]        


def getresult(fn):
    result = []
    with open(fn, "r") as f:
        l = f.readline()
        while l:
            l = l.strip().split()
            for i in range(len(l)):
                l[i] = float(l[i])
            l = softmax(l)
            result += [l]
            l = f.readline()
    return result
results = []


results += [getresult("output/roberta_wwm_ext_large_gbc3_1epoch_666infer/logits_dev.txt") + \
            getresult("output/roberta_wwm_ext_large_gbc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_lbc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_ibc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_ctc3_1epoch_666infer/logits_test.txt")]

results += [getresult("output/roberta_wwm_ext_large_lbc3_1epoch_666infer/logits_dev.txt") + \
            getresult("output/roberta_wwm_ext_large_gbc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_lbc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_ibc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_ctc3_1epoch_666infer/logits_test.txt")]

results += [getresult("output/roberta_wwm_ext_large_ibc3_1epoch_666infer/logits_dev.txt") + \
            getresult("output/roberta_wwm_ext_large_gbc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_lbc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_ibc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_ctc3_1epoch_666infer/logits_test.txt")]

results += [getresult("output/roberta_wwm_ext_large_ctc3_1epoch_666infer/logits_dev.txt") + \
            getresult("output/roberta_wwm_ext_large_gbc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_lbc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_ibc3_1epoch_666infer/logits_test.txt") + \
            getresult("output/roberta_wwm_ext_large_ctc3_1epoch_666infer/logits_test.txt")]


print(len(results[0]))

result = []
for i in range(len(data)):
    for j in range(len(data[i][1])):
        answer = []
        for k in range(len(data[i][1][j]["choice"])):
            if data[i][1][j]["choice"][k] == data[i][1][j]["answer"]:
                answer += [1]
            else:
                answer += [0]
        for k in range(len(data[i][1][j]["choice"]), 6):
            answer += [0]
        result += [answer]
results += [result]

assert(len(results) == len(weights))

result = []
for i in range(len(results[0])):
    result += [[0] * len(results[0][0])]

for i in range(len(results)):
    for j in range(len(results[i])):
        for k in range(len(results[i][j])):
            result[j][k] += weights[i][j] * results[i][j][k]

for i in range(len(result)):
    if abs(sum(result[i])-1) > 1e-5:
        print("warning:", result[i])
            
k = 0
acc, all = 0, 0
for i in range(len(data)):
    for j in range(len(data[i][1])):
        data[i][1][j]["kl"] = result[k]
        pred = 0
        for l in range(1, nchoice[k]):
            if result[k][l] > result[k][pred]:
                pred = l
        if data[i][1][j]["choice"][pred] == data[i][1][j]["answer"]:
            acc += 1
        all += 1
        k += 1

print(acc/all)


parent_dir = "data/"

directory_c3 = "c3_kl0.5"
path_c3 = os.path.join(parent_dir, directory_c3)
os.mkdir(path_c3)
print("Directory '% s' created" % directory_c3)

directory_weak = "scriptc3_kl0.5"
path_weak = os.path.join(parent_dir, directory_weak)
os.mkdir(path_weak)
print("Directory '% s' created" % directory_weak)


with open(path_weak + "/" + "script_kl0.5.json", "w", encoding='utf8') as f:
    json.dump(data, f, indent=1, ensure_ascii=False)


with open(path_c3 + "/" + "c3_release_script-kl0.5-1s.json", "w", encoding='utf8') as f:
    json.dump(data[:split0], f, indent=1, ensure_ascii=False)
