import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from random import randint

from torch.utils.data import Dataset

def basic(data, tokenizer):
    formatted = []
    max_length = 300
    tokenizer.pad_token = tokenizer.eos_token

    for _, row in tqdm(data.iterrows()):
        for q_col in ['질문_1', '질문_2']:
            for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
                input_text = row[q_col] + tokenizer.eos_token + row[a_col]
                input_ids = tokenizer.encode(input_text, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
                formatted.append(input_ids)
        
    formatted_data = torch.cat(formatted, dim=0)
    
    return formatted

def get_df(data):
    # 빈 리스트를 만들어 질문과 답변 쌍을 저장합니다.
    pairs = []

    for _, row in data.iterrows():
        for q_col in ['질문_1', '질문_2']:
            question = row[q_col]
            for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
                answer = row[a_col]
                pairs.append([question, answer])

    # 데이터프레임을 생성합니다.
    df = pd.DataFrame(pairs, columns=['질문', '답변'])
    return df



class Hansol_Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=300, transforms=None):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        if transforms:
            self.transform = transforms
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx]['질문']
        answer = self.data.iloc[idx]['답변']
        input_text = question + self.tokenizer.eos_token + answer
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt')
        
        input_ids = torch.nn.functional.pad(input_ids, (0, self.max_length - input_ids.size(1)))
        
        return input_ids
    
