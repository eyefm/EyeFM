# Clinically Translatable Eyecare Foundational Model: Design, Validation and Randomized Controlled Trial

## Contents

1. Requirements
2. Prepare the environment
3. Finetuning of vision module
4. Pretraining of vision module
5. Pretraining of vision-language

## Requirements

This software requires a **Linux** system: [**Ubuntu 22.04**](https://ubuntu.com/download/desktop) or  [**Ubuntu 20.04**](https://ubuntu.com/download/desktop) (other versions are not tested)   and  [**Python3.8**](https://www.python.org) (other versions are not tested).

The **Python packages** needed are listed below. They can also be found in `reqirements.txt`.

```
torch>=1.10.0
torchvision>=0.11.1
timm>=0.4.12
einops>=0.3.2
pandas>=1.3.4
albumentations>=1.1.0
wandb>=0.12.11
flash-attn==2.5.8,
ninja>=1.11.1.0
open-clip-torch>=2.11.0
```

## Prepare the environment

### 1. Download the project

- Open the terminal in the system.

- Clone this repo file to the home path.

```
git clone https://github.com/eyefm/EyeFM.git
```

- Change the current directory to the project directory

```
cd EyeFM
```

### 2. Prepare the running environment
- Install Pytorch 1.10.0 (Cuda 11.1)

```
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

- Install other Python packages

```
pip install -r requirements.txt
```

## Fine-tuning of vision module

### 1. Prepare the datasets

During fine-tuning, images and labels are stored in CSV files: `train.csv`, `val.csv`, and `test.csv`, corresponding to the training, validation, and testing sets, respectively. These files should be placed in the same directory. In each CSV file, `cfp` and `oct` refer to the image paths of the corresponding modalities, while `label` denotes the associated ground truth. For paired-modality fine-tuning, each row contains paired data from the two or more modalities.
```
├── fine-tuning csv path
    ├──train.csv
    ├──val.csv
    ├──test.csv
```
```
├── fine-tuning image path
    ├──cfp
    	├──ffds.png
    	fs1fsd.png
    ├──oct
    	├──gdfg.png
    	├──d4fsad.png
```
|  cfp   | oct   | label |
|  :--:  | :--:  |  :--:|
|  cfp/ffds.png | oct/gdfg.png |0|
|  cfp/fs1fsd.png  | oct/d4fsad.png |1|

### 2. Configuration of fine-tuning workflows
To fine-tune EyeFM on downstream vision tasks, run:
```
cd image_module
python run_finetuning.py --config cfgs/finetune/finetune.yaml --finetune /path/to/checkpoint --in_domains cfp --epochs 50 --batch_size 16 --blr 0.0005 --input_size 224 --data_path path/to/fine-tuning/image --csv_path path/to/fine-tuning/csv/file
```
- The fine-tuning scripts support both YAML configuration files (e.g., [finetune.yaml](https://github.com/eyefm/EyeFM/blob/main/image_module/cfgs/finetune/finetune.yaml)) and command-line arguments.
- Arguments specified in the configuration file override the default settings, while command-line arguments override both the default and configuration arguments.
### 3. Results interpretation
Given an input image, the model will output the probability distribution across different disease categories. For example, when provided with an image of diabetic retinopathy (DR), the model will output the probabilities corresponding to each of the five severity levels.

<img src="https://github.com/eyefm/EyeFM/blob/main/image_module/data/Finetune data for Experiment 1&2/DR/Severe retinopathy.png" alt="Severe retinopathy" style="zoom:50%;" />

```
No retinopathy:0.0
Mild retinopathy:0.012
Moderate retinopathy:0.121
Severe retinopathy:0.753
Proliferative retinopathy:0.114
```

## Pretraining of vision module
### 1. Prepare the datasets

During the pretraining phase, images are stored in a CSV file named`train.csv`, as shown in the table below. In this file, the column headers `cfp`, `oct`, `eyephoto`, `uwf`, and `ffa` correspond to different image modalities: Color Fundus Photography (CFP), Optical Coherence Tomography (OCT), External Eye Photo (EEP), Ultra-Widefield Fundus (UWF), and Fundus Fluorescein Angiography (FFA). Each entry under these columns specifies the corresponding image name.

|  cfp   | oct   | eyephoto | uwf | ffa|
|  :--:  | :--:  |  :--:|:--:|:--:|
| cfp/dsg.png | oct/hsdg.png |eyephoto/gsag.png|uwf/shhsd.png|ffa/iqpakg.png|
|  cfp/hsdjt.png  | oct/asd5fh.png |eyephoto/sd7dslg.png|uwf/lajg.png|ffa/glmz9h.png|

### 2. Configuration of pretraining workflows
To complete the pretraining of EyeFM, run:

```
cd image_module
python run_pretraining.py --config cfgs/pretrain/pretraining.yaml --epochs 800 --batch_size 100 --blr 0.0001 --input_size 224 --data_path path/to/pretraining/image --csv_path path/to/pretraining/csv
```

- The pretraining scripts support both YAML configuration files (e.g., [pretraining.yaml](https://github.com/eyefm/EyeFM/blob/main/image_module/cfgs/pretrain/pretraining.yaml)) and command-line arguments.
- Arguments specified in the configuration file override the default settings, while command-line arguments override both the default and configuration arguments.


## Pretraining of vision-language

After the pretraining of vision module, you can start vision-language joint training.

### Preparing dataset
Images and texts are stored in JSON files: `train.json`, `val.json`, and `test.json`, corresponding to the training, validation, and testing sets, respectively. 

These files should be placed in the same directory. In each JSON file, `image` refer to the image paths, while `conversations` denotes the related text. 
```
{
    "id": "FFA1",
    "image": "FFA1.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "What is the expression of the image\n<image>"
      },
      {
        "from": "ophthalmologist",
        "value": "The fluorescein angiography of the left eye shows early microvascular aneurysm-like proliferation starting at the posterior pole and peripheral regions. As the angiography progresses, fluorescein leakage gradually appears, with a cluster of high fluorescence observed at the branch of the temporal artery, followed by rapid fluorescein leakage. A large number of laser spots are seen from the posterior pole to the temporal peripheral area, with gradual fluorescein staining. In the later stages, a patchy, blurry high fluorescence is observed in the macular region, and the fluorescence of the optic disc gradually increases, with the boundary becoming discernible."
      }
    ]
  }
```

### 1. Initialization
To ensure the smooth adaptation, you can:

1. Obtain the original LLaVA-7B weight following the instructions [here](https://llava-vl.github.io/).
2. Obtain the original LLaMA-7B weight following the instructions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
3. Utilize the weights of the visual module obtained from the above steps.


### 2. Ophthalmic concept alignment

This part is trained on 8 A800 GPUs with 40GB memory with the follow code:

```
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \ 
train/train_mem.py --model_name_or_path /EyeFM/cn_llava  --data_path /EyeFM/ours_cn_all.json --image_folder / --tune_mm_mlp_adapter True --output_dir /EyeFM/output --vision_tower /EyeFM/transfer_image_encoder --mm_vision_select_layer -2 --mm_use_im_start_end True --num_train_epochs 5 --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 3 --learning_rate 2e-3 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --model_max_length 1024 --lazy_preprocess True --gradient_checkpointing True --dataloader_num_workers 8 --report_to none --bf16 false --tf32 false
```

### 3. Ophthalmic visual instruct tuning

This part is trained on 8 A800 GPUs with 40GB memory with the follow code:

```
torchrun --nnodes=1 --nproc_per_node=8 --master_port=25001 \
train/train_mem.py --model_name_or_path /EyeFM/cn_llava  --data_path /EyeFM/ours_cn_all.json --image_folder / --tune_mm_mlp_adapter True --output_dir /EyeFM/output --vision_tower /EyeFM/transfer_image_encoder --mm_vision_select_layer -2 --mm_use_im_start_end True --num_train_epochs 3 --per_device_train_batch_size 1 --per_device_eval_batch_size 4 --gradient_accumulation_steps 8 --evaluation_strategy "no" --save_strategy "steps" --save_steps 1000 --save_total_limit 3 --learning_rate 2e-5 --weight_decay 0. --warmup_ratio 0.03 --lr_scheduler_type "cosine" --logging_steps 1 --model_max_length 1024 --lazy_preprocess True --gradient_checkpointing True --dataloader_num_workers 8 --report_to none --bf16 false --tf32 false
```

### 4. LoRA 

If you are do not have enough GPU memory, you can use LoRA following the intructions [here](https://github.com/haotian-liu/LLaVA?tab=readme-ov-file#train)

### 5. Evaluation
We evaluate models on the report generation and VQA tasks using `Meteor` and `ROUNGE`. The details of the tasks are shown in our paper. 

