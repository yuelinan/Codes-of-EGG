
import os
import argparse
import random
from tqdm import tqdm, trange
import csv
import glob 
import json
from sklearn import metrics
import numpy as np
import torch
import os
import torch.nn as nn
from utils import *
from model_bart import *
from torch.utils.data import TensorDataset, DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import prettytable as pt
from sacrebleu.metrics import BLEU

parser = argparse.ArgumentParser(description='VMask classificer')
# batch 128, gpu 10000M
parser.add_argument('--aspect', type=int, default=0, help='aspect')
parser.add_argument('--dataset', type=str, default='beer')
parser.add_argument('--model_name', type=str, default='RNP', help='model name')
parser.add_argument('--lr', type=float, default=5e-5, help='initial learning rate')
parser.add_argument('--max_len', type=int, default=300, help='max_len')
parser.add_argument('--weight_decay', default=0, type=float, help='adding l2 regularization')
parser.add_argument('--dropout', type=float, default=0.2, help='the probability for dropout')
parser.add_argument('--alpha_rationle', type=float, default=0.2, help='alpha_rationle')
parser.add_argument('--hidden_dim', type=int, default=768, help='number of hidden dimension')
parser.add_argument("--activation", type=str, dest="activation", default="tanh", help='the choice of \
        non-linearity transfer function')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--types', type=str, default='legal', help='data_type')
parser.add_argument('--epochs', type=int, default=16, help='number of epochs for training')
parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
parser.add_argument('--class_num', type=int, default=2, help='class_num')
parser.add_argument('--save_path', type=str, default='', help='save_path')
parser.add_argument('--date', type=str, default='1207', help='save_path')
parser.add_argument('--max_target_length', type=int, default=15, help='save_path')


args = parser.parse_args()
import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

def random_seed():
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return

args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(args.device)
process = Event_Process(args)



for k, v in vars(args).items():
    logger.info("{:20} : {:10}".format(k, str(v)))

def get_charge_name(text):
    text_list = re.findall(".{1}",text)
    new_text = " ".join(text_list)
    return new_text+' 罪。'
def main():
    
    model = eval(args.model_name).from_pretrained("fnlp/bart-base-chinese",return_dict=True)
    
    data_all = process.process_data(args.types, model)
    print(len(data_all))
    dataloader = DataLoader(data_all, batch_size = args.batch_size, shuffle=True, num_workers=0, drop_last=False)

    test_data_all = process.process_data('test',model)
    print(len(test_data_all))
    test_dataloader = DataLoader(test_data_all, batch_size = args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=args.lr)   

    total_steps = len(data_all) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    model.train()

    for epoch in range(1, args.epochs+1):
        model.train()

        logger.info("Trianing Epoch: {}/{}".format(epoch, int(args.epochs)))
        for step,batch in enumerate(tqdm(dataloader)):
            # if step>2:break
            batch = tuple(t.to(args.device) for t in batch) 
            
            input_ids,attention_mask,target_input_ids,decoder_input_ids,event_input_ids,event_attention_mask = batch
            optimizer.zero_grad()
            
            adc_output = model(input_ids = input_ids, attention_mask = attention_mask, labels = target_input_ids, decoder_input_ids = decoder_input_ids, event_id = event_input_ids, event_attention_mask=event_attention_mask)
            
            loss = adc_output.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()


        if epoch>0:
            model.eval()
            bleu = BLEU()
            print('test')
            ##############################  test  ##############################
            preds, candicate = [], []
            predictions_charge = []
            true_charge = []
            for step,batch in enumerate(tqdm(test_dataloader)):
                batch = tuple(t.to(args.device) for t in batch) 
                input_ids,attention_mask,target_input_ids,decoder_input_ids,event_input_ids,event_attention_mask = batch
                
                with torch.no_grad():
                    # decode sc
                    # decode adc and charge

                    generated_tokens = model.generate(input_ids = input_ids,max_length=100,num_beams=5).cpu().numpy()
                    
                label_tokens = target_input_ids.cpu().numpy()   

                view_decoded_preds = process.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                
                
                decoded_preds = [view_decoded_preds[i] for i in range(len(view_decoded_preds))]
                
                # label_tokens = np.where(label_tokens != -100, label_tokens, process.tokenizer.pad_token_id)
                decoded_labels = process.tokenizer.batch_decode(label_tokens, skip_special_tokens=True)
                preds += [pred.strip() for pred in decoded_preds]
                candicate += [[label.strip()] for label in decoded_labels]
                
            # bleu_score = bleu.corpus_score(preds, candicate).score
            if epoch>0:
                path = args.model_name.lower()+ str(epoch) + '.json'
                f_result = open(path,'a+')
                result = {}
                result['preds'] = preds
                result['candicate'] = candicate
                json_str = json.dumps(result, ensure_ascii=False)
                f_result.write(json_str)
                f_result.write('\n')



if __name__ == "__main__":
    random_seed()
    criterion = nn.CrossEntropyLoss()
    main()
