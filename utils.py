import json
import torch
import re
import numpy as np
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader

def relaw(rel, text):
    p = re.compile(rel)
    m = p.search(text)
    if m is None:
        return None
    return m.group(0)

def re_view(xx):
    re_artrule = r'本院 认为(.*)其 行为'
    art = relaw(re_artrule, xx)
    if art == None:
        return -1
    return art[12:-4]

def make_kv_string(d):
    out = []
    for k, v in d.items():
        if isinstance(v, float):
            out.append("{} {:.4f}".format(k, v))
        else:
            out.append("{} {}".format(k, v))

    return " ".join(out)

def get_value(res):
    if res["TP"] == 0:
        if res["FP"] == 0 and res["FN"] == 0:
            precision = 1.0
            recall = 1.0
            f1 = 1.0
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
    else:
        precision = 1.0 * res["TP"] / (res["TP"] + res["FP"])
        recall = 1.0 * res["TP"] / (res["TP"] + res["FN"])
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def gen_result(res, test=False, file_path=None, class_name=None):
    precision = []
    recall = []
    f1 = []
    total = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for a in range(0, len(res)):
        total["TP"] += res[a]["TP"]
        total["FP"] += res[a]["FP"]
        total["FN"] += res[a]["FN"]
        total["TN"] += res[a]["TN"]

        p, r, f = get_value(res[a])
        precision.append(p)
        recall.append(r)
        f1.append(f)

    micro_precision, micro_recall, micro_f1 = get_value(total)

    macro_precision = 0
    macro_recall = 0
    macro_f1 = 0
    for a in range(0, len(f1)):
        macro_precision += precision[a]
        macro_recall += recall[a]
        macro_f1 += f1[a]

    macro_precision /= len(f1)
    macro_recall /= len(f1)
    macro_f1 /= len(f1)

    #print("Micro precision\t%.4f" % micro_precision)
    #print("Micro recall\t%.4f" % micro_recall)
    print("Micro f1\t%.4f" % micro_f1)
    print("Macro precision\t%.4f" % macro_precision)
    print("Macro recall\t%.4f" % macro_recall) 
    print("Macro f1\t%.4f" % macro_f1)

    return micro_f1, macro_precision, macro_recall,macro_f1

def eval_data_types(target,prediction,num_labels):
    ground_truth_v2 = []
    predictions_v2 = []
    for i in target:
        v = [0 for j in range(num_labels)]
        v[i] = 1
        ground_truth_v2.append(v)
    for i in prediction:
        v = [0 for j in range(num_labels)]
        v[i] = 1
        predictions_v2.append(v)

    res = []
    for i in range(num_labels):
        res.append({"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    y_true = np.array(ground_truth_v2)
    y_pred = np.array(predictions_v2)
    for i in range(num_labels):
    
        outputs1 = y_pred[:, i]
        labels1 = y_true[:, i] 
        res[i]["TP"] += int((labels1 * outputs1).sum())
        res[i]["FN"] += int((labels1 * (1 - outputs1)).sum())
        res[i]["FP"] += int(((1 - labels1) * outputs1).sum())
        res[i]["TN"] += int(((1 - labels1) * (1 - outputs1)).sum())
    micro_f1, macro_precision, macro_recall,macro_f1 = gen_result(res)

    return micro_f1, macro_precision, macro_recall,macro_f1


class Event_Process():
    def __init__(self,args):
        
        self.tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
        self.f_event_train = 'train.json'
        self.f_event_test = 'test.json'
        
        self.max_len = args.max_len
    def encode_fn(self,text_list):	
        tokenizer = self.tokenizer.batch_encode_plus(
            text_list,
            padding = True,
            truncation = True,
            max_length = self.max_len,
            return_tensors='pt' 
        )
        input_ids = tokenizer['input_ids']
        attention_mask = tokenizer['attention_mask']
        return input_ids,attention_mask

    def process_data(self,types,model):
        if types == 'train':
            event_path = self.f_event_train
        else:
            event_path = self.f_event_test
        events = []
        fact_source = []
        charge_label = [] 
        view_all = []

        f_event = open(event_path,'r',encoding='utf8')
        
        for index,line in enumerate(f_event):
            line = json.loads(line)
            event = line['event']
            view = line['view']
            fact = line['fact'][0:-7]
            fact = event + fact
            fact_source.append(fact)
            view_all.append(view)
            events.append(event)
        print('event')
        event_input_ids,event_attention_mask = self.encode_fn(events)
        print('input_ids')
        input_ids,attention_mask = self.encode_fn(fact_source)
        print('target_input_ids')
        target_input_ids,_ = self.encode_fn(view_all)
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(target_input_ids)
    
        data = TensorDataset(input_ids,attention_mask,target_input_ids,decoder_input_ids,event_input_ids,event_attention_mask)
 
        return data
    


    def process_data_free(self,types,model):
        if types == 'train':
            event_path = self.f_event_train
        else:
            event_path = self.f_event_test
        events = []
        fact_source = []
        charge_label = [] 
        view_all = []

        f_event = open(event_path,'r',encoding='utf8')
        
        for index,line in enumerate(f_event):
            line = json.loads(line)
            event = line['event']
            view = line['view']
            fact = line['fact'][0:-7]
            fact_source.append(fact)
            view_all.append(view)
            events.append(event)
        print('event')
        event_input_ids,event_attention_mask = self.encode_fn(events)
        print('input_ids')
        input_ids,attention_mask = self.encode_fn(fact_source)
        print('target_input_ids')
        target_input_ids,_ = self.encode_fn(view_all)
        decoder_input_ids = model.prepare_decoder_input_ids_from_labels(target_input_ids)
    


        data = TensorDataset(input_ids,attention_mask,target_input_ids,decoder_input_ids,event_input_ids,event_attention_mask)
 
        return data
    
