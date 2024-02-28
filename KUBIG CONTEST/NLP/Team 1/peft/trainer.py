
from tqdm import tqdm
import torch

def train(config, model, optimizer, tokenizer, train_dataloader, device, train_sampler=None):
    model.train()

    # 모델 학습
    for epoch in range(config['EPOCHS']):
        if config['ddp']:
                train_sampler.set_epoch(epoch)
        total_loss = 0
        progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
        for batch_idx, batch in enumerate(progress_bar):
            # 데이터를 GPU단으로 이동
            batch = torch.squeeze(batch)
            batch = batch.to(device)
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # 진행률 표시줄에 평균 손실 업데이트
            progress_bar.set_description(f"Epoch {epoch+1} - Avg Loss: {total_loss / (batch_idx+1):.4f}")

        # 에폭의 평균 손실을 출력
        print(f"Epoch {epoch+1}/{config['EPOCHS']}, Average Loss: {total_loss / len(train_dataloader)}")
        # 5 에폭마다 모델과 토크나이저 저장
        if config['ddp']:
            if (epoch+1) % 1 == 0:
                model.module.save_pretrained(f"/home/thdtjd06/checkpoints/llama2_0223_epoch{epoch+1}")
                tokenizer.save_pretrained(f"/home/thdtjd06/checkpoints/llama2_0223_epoch{epoch+1}")
        else:
            if (epoch+1) % 1 == 0:
                model.save_pretrained(f"/home/thdtjd06/checkpoints/llama2_0223_epoch{epoch+1}")
                tokenizer.save_pretrained(f"/home/thdtjd06/checkpoints/llama2_0223_epoch{epoch+1}")
            
    return model, tokenizer
            
            
def test(config, model, test, tokenizer, device):
    # test.csv의 '질문'에 대한 '답변'을 저장할 리스트
    preds = []

    # '질문' 컬럼의 각 질문에 대해 답변 생성
    for test_question in tqdm(test['질문']):
        # 입력 텍스트를 토큰화하고 모델 입력 형태로 변환
        input_ids = tokenizer.encode(test_question + tokenizer.eos_token, return_tensors='pt')

        # 답변 생성
        with torch.no_grad():
            output_sequences = model.generate(
                input_ids=input_ids.to(device),
                max_length=300,
                temperature=0.9,
                top_k=1,
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1
            )

        # 생성된 텍스트(답변) 저장
        for generated_sequence in output_sequences:
            full_text = tokenizer.decode(generated_sequence, skip_special_tokens=False)
            # 질문과 답변의 사이를 나타내는 eos_token (</s>)를 찾아, 이후부터 출력
            answer_start = full_text.find(tokenizer.eos_token) + len(tokenizer.eos_token)
            answer_only = full_text[answer_start:].strip()
            answer_only = answer_only.replace('\n', ' ')
            answer_only = answer_only.replace('<s>', '') #추가
            preds.append(answer_only)
            
    return preds