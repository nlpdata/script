
Improving Machine Reading Comprehension with Contextualized Commonsense Knowledge
=====

This repository maintains the code and resource for the above ACL'22 paper. Please contact script@dataset.org if you have any questions or suggestions.

[Paper](https://aclanthology.org/2022.acl-long.598/):
```
@inproceedings{sun-2022-improving,
    title = "Improving Machine Reading Comprehension with Contextualized Commonsense Knowledge",
    author = "Sun, Kai  and
      Yu, Dian  and
      Chen, Jianshu  and
      Yu, Dong  and
      Cardie, Claire",
    booktitle = "Proceedings of the ACL 2022",
    year = "2022",
    address = "Dublin, Ireland",
    url = "https://aclanthology.org/2022.acl-long.598",
    pages = "8736--8747",
}
```

Files in this repository:

* ```data/en/en_b.json```: weakly-labeled English MRC instances constructed based on pattern B_c. 
* ```data/en/en_i.json```: weakly-labeled English MRC instances constructed based on pattern I. 
* ```data/en/en_o.json```: weakly-labeled English MRC instances constructed based on pattern O. 
* ```data/cn/lb/cat_{lb1,lb2}.json```: samples of the weakly-labeled Chinese MRC instances constructed by B_c. 
* ```data/cn/gb/cat_{gb1,gb2}.json```: samples of the weakly-labeled Chinese MRC instances constructed by B_n. 
* ```data/cn/ib/cat_{ib1,ib2}.json```: samples of the weakly-labeled Chinese MRC instances constructed by I. 
* ```data/cn/ct/cat_{ct1,ct2}.json```: samples of the weakly-labeled Chinese MRC instances constructed by O. 
* ```data/c3_soft/c3_train_soft.json```: soft labels of the C3 training data used for fine-tuning student models in the multi-teacher paradigm. 


Due to the copyright issues, full weakly-labeled Chinese MRC instances are not provided. We use the Englsih scripts from the [ScriptBase Corpus](https://github.com/EdinburghNLP/scriptbase). As almost all scripts are written following the standard templates, using patterns B_n can hardly extract any knowledge triples. To use contextualized knowledge (i.e., (verbal, context, nonverbal) triples) for non-MRC tasks, you can just use (question, document, answer) in MRC instances.

The data format is as follows.
```
[
  [
    [
      document 1
    ],
    [
      {
        "question": document 1 / question 1,
        "choice": [
          document 1 / question 1 / answer option 1,
          document 1 / question 1 / answer option 2,
          ...
        ],
        "answer": document 1 / question 1 / correct answer option
      }
    ],
    document 1 / question 1 / id
  ],
  [
    [
      document 2
    ],
    [
      {
        "question": document 2 / question 1,
        "choice": [
          document 2 / question 1 / answer option 1,
          document 2 / question 1 / answer option 2,
          ...
        ],
        "answer": document 2 / question 1 / correct answer option
      }
    ],
    document 2 / question 1 / id
  ],
  ...
]
```

**Experiments**
=====

**STEP I: Train four teacher models**

Set the file paths for the pre-trained language model [RoBERTa-wwm-ext-large](https://github.com/ymcui/Chinese-BERT-wwm) (PyTorch version), [C3](https://github.com/nlpdata/c3), and output folder in ```run-teacher.sh``` and execute
	
```
bash run-teacher.sh
```

**STEP II: Generate soft lables for both weakly-labled and clean data**

```
bash run-infer.sh
```

Based on the resulting four folders, execute the following command:

```
python kltrainss_script.py
```

**STEP III: Train a student model**

```
bash run-teacher-kl.sh
```

**STEP IV: Fine-tune the student model on the downstream MRC data**

```
bash run-student-kl.sh
```

**Reproducibility**
=====
Download the [model](https://share.weiyun.com/j1hovV0E) that is pretrained on the combination of soft weakly-labeled data and soft clean data. Execute the following command (set the path first):

```
bash run-student-kl-acl.sh
```

The code has been tested with Python 3.6 and PyTorch 1.1.


**Disclaimer**
=====
This is not an officially supported Tencent product.

