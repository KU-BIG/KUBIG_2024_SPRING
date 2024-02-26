import torch
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from modules.utils import *
import wandb
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


class BaiscTrainer:
    def __init__(self, CFG, model, train_loader, valid_loader):
        self.CFG = CFG
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = CFG["DEVICE"]
        self.epochs = CFG["TRAIN"]["EPOCHS"]
        self.es_patient = CFG["TRAIN"]["EARLY_STOPPING"]
        self.gradient_accumulation_steps = CFG["TRAIN"]["ACCUMUL_STEPS"]
        self.optimizer = get_optimizer(self.CFG)
        self.scheduler = get_scheduler()

    def get_optimizer(self):
        select_optimizer = self.CFG["TRAIN"]["OPTIMIZER"]
        learning_rate = self.CFG["TRAIN"]["LEARNING_RATE"]
        if select_optimizer.lower() == "adamw":
            optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        return optimizer

    def get_scheduler(self):
        select_scheduler = self.CFG["TRAIN"]["SCHEDULER"]
        select_scheduler_cfg = select_scheduler["CFG"]
        if select_scheduler["NAME"].lower() == "cosineannealinglr":
            scheduler = CosineAnnealingLR(
                self.optimizer, T_max=select_scheduler_cfg["TMAX"]
            )
        return scheduler

    def train_epoch(self):
        print(f"..Epoch {self.current_epoch+1}/{self.epochs}..")
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0
        progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        for batch_idx, (input_ids, attention_mask) in progress_bar:
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            outputs = self.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
            )
            loss = outputs.loss / self.gradient_accumulation_steps
            loss.backward()
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(self.train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()
            total_loss += loss.item() * self.gradient_accumulation_steps
        return total_loss / len(self.train_loader)

    def validate(self):
        total_loss = 0
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(
                enumerate(self.valid_loader), total=len(self.valid_loader)
            )
            for batch_idx, (input_ids, attention_mask) in progress_bar:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                outputs = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=input_ids
                )
                loss = outputs.loss
                total_loss += loss.item()
        return total_loss / len(self.valid_loader)

    def train(self):
        best_loss = float("inf")
        es_count = 1
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()
            valid_loss = self.validate()
            self.scheduler.step()

            if valid_loss < best_loss:
                es_count = 1
                best_loss = valid_loss
                print("Best Loss Updated. New Best Model Saved.")
                self.save_model("model")
            else:
                print(f"Eearly Stopping Count: {es_count}/{self.es_patient}")
                es_count += 1

            if es_count >= self.es_patient:
                print(
                    f"Early stopping patience {self.es_patient} has been reached, validation loss has not improved, ending training."
                )
                break

            print(f"Train Loss: {train_loss}, Valid Loss: {valid_loss}")
            wandb.log({"Train loss": train_loss, "Valid Loss": valid_loss}, step=epoch)


class HFTraining:
    def __init__(self, CFG) -> None:
        self.CFG = CFG
        self.train_cfg = CFG["TRAIN"]
        self.training_args = TrainingArguments(
            seed=CFG["SEED"],
            output_dir=f'{CFG["SAVE_PATH"]}/{CFG["NAME"]}_{CFG["START_TIME"]}',
            num_train_epochs=self.train_cfg["EPOCHS"],
            per_device_train_batch_size=self.train_cfg["BATCH_SIZE"],
            per_device_eval_batch_size=self.train_cfg["BATCH_SIZE"],
            gradient_accumulation_steps=self.train_cfg["ACCUMUL_STEPS"],
            learning_rate=self.train_cfg["LEARNING_RATE"],
            optim="adamw_torch",
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            save_strategy="epoch",
            logging_dir="./logs",
            evaluation_strategy="epoch",
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1e-8,
            lr_scheduler_type="cosine",
            warmup_steps=10,
            load_best_model_at_end=True,
            report_to=["wandb"],
            run_name=f"{self.CFG['NAME']}_{self.CFG['START_TIME']}",
            group_by_length=True,
        )

    def run(self, model, train_dataset, eval_dataset):
        trainer = Trainer(
            model=model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.train_cfg["EARLY_STOPPING"]
                )
            ],
        )
        trainer.train()
        trainer.save_model(
            f'{self.CFG["SAVE_PATH"]}/{self.CFG["NAME"]}_{self.CFG["START_TIME"]}/best_model'
        )
        return trainer
