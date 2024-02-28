import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast
from konlpy.tag import Mecab
from random import randint

from torch.utils.data import Dataset


# 문장 셔플 및 조사 삭제 함수 정의
def shuffle_and_delete_josa(sentence):
    mecab = Mecab()
    # 문장 셔플
    new_sentences = []

    if len(sentence.split()) == 1:
        new_sentences.append("")
    elif len(sentence.split()) == 2:
        new_sentences.append(sentence.split()[1] + " " + sentence.replace(sentence.split()[1], ""))
    elif len(sentence.split()) == 3:
        new_sentences.append(sentence.split()[1] + " " + sentence.replace(sentence.split()[1], ""))
        new_sentences.append(sentence.split()[2] + " " + sentence.replace(sentence.split()[2], ""))
    else:
        new_sentences.append(sentence.split()[1] + " " + sentence.replace(sentence.split()[1], ""))
        new_sentences.append(sentence.split()[2] + " " + sentence.replace(sentence.split()[2], ""))
        new_sentences.append(sentence.split()[len(sentence.split()) - 1] + " " + sentence.replace(
            sentence.split()[len(sentence.split()) - 1], ""))

    shuffled_sentence = new_sentences[randint(0, len(new_sentences) - 1)]

    # 조사 삭제
    tagged_sentence = mecab.pos(shuffled_sentence)
    new_sentence = [word for word, tag in tagged_sentence if 'Josa' not in tag]
    deleted_josa_sentence = ' '.join(new_sentence)

    return shuffled_sentence, deleted_josa_sentence

def get_df(data, aug=False):
    # 빈 리스트를 만들어 질문과 답변 쌍을 저장합니다.
    pairs = []

    for _, row in data.iterrows():
        for q_col in ['질문_1', '질문_2']:
            question = row[q_col]
            for a_col in ['답변_1', '답변_2', '답변_3', '답변_4', '답변_5']:
                answer = row[a_col]
                pairs.append([question, answer])
                
                if aug:
                    shuffled_text, deleted_josa_text = shuffle_and_delete_josa(row[a_col])

                    # 셔플한 문장을 데이터셋에 추가
                    pairs.append([question, shuffled_text])

                    # 조사 삭제한 문장을 데이터셋에 추가
                    pairs.append([question, deleted_josa_text])

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
    
