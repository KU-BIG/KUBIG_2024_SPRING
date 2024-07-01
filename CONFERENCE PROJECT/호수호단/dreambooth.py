#terminal
git clone https://github.com/huggingface/diffusers
pip install -U -r diffusers/examples/dreambooth/requirements.txt
cd diffusers
pip install -e .
accelerate config default


export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="path_to_training_images"
export CLASS_DIR="path_to_class_images"
export OUTPUT_DIR="path_to_saved_model"

cd "C:/Users/tydbs/OneDrive/바탕 화면/학교 활동/2023~.KUBIG/2024.cv/code"
accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --class_data_dir=$CLASS_DIR \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --instance_prompt="a photo of sks dog" \
  --class_prompt="a photo of dog" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --max_train_steps=800