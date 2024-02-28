
import pandas as pd
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast, AdamW
from data import *
import os
from trainer import train, test
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.distributed as dist


from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
        
    config = {
        'LR' : 2e-5, # Learning Rate
        'EPOCHS' : 30, # 학습 Epoch
        'ddp' : True,
        'aug' : False,
        'inference' : True
    }
    
    # ddp setting 
    if config['ddp']:
        config['rank']= int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(config['rank'])
        torch.cuda.empty_cache()
        config['world_size'] =  int(os.environ['WORLD_SIZE'])
        dist.init_process_group(backend='gloo', #ncll
                                init_method='env://',
                                world_size=config['world_size'],
                                rank=config['rank'])
        config['world_size'] = float(dist.get_world_size())
    else: 
        config['rank'] = 0
        config['world_size'] = 1
        
    if config['ddp']:
        device = torch.device('cuda:{}'.format(config['rank']))
    else:
        device = 'cuda'
    print(device)

    # tokenizer, dataloader
    model_dir = 'skt/kogpt2-base-v2'
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir , eos_token='</s>')
    model = GPT2LMHeadModel.from_pretrained(model_dir)
        
    #name = 'heegyu/WizardVicuna-open-llama-3b-v2'
    #tokenizer = AutoTokenizer.from_pretrained(name)
    #model = AutoModelForCausalLM.from_pretrained(name)
    model.to(device) # 모델을 GPU단으로 이동
    
    if config['ddp']:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[config['rank']], find_unused_parameters=True)
        
        
    if config['inference']:
        test_data = pd.read_csv('./data/test.csv')

        preds = test(config, model, test_data, tokenizer, device)

        emb_model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

        # 생성한 모든 응답(답변)으로부터 Embedding Vector 추출
        pred_embeddings = emb_model.encode(preds)
        submit = pd.read_csv('./data/sample_submission.csv')
        # 제출 양식 파일(sample_submission.csv)을 활용하여 Embedding Vector로 변환한 결과를 삽입
        submit.iloc[:,1:] = pred_embeddings
        submit.to_csv('./code_submit_epoch30.csv', index=False)
        
    else: 
        data = pd.read_csv('./data/train.csv')
        df = get_df(data, aug=config['aug'])
        train_data = Hansol_Dataset(df, tokenizer)
        if config['ddp']:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
                                                                            rank=config['rank'])
        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)


        # 모델 학습 설정
        optimizer = AdamW(model.parameters(), lr=config['LR'])


        #model train
        model.train()
        model, tokenizer = train(config, model, optimizer, tokenizer, train_dataloader, device, train_sampler)
