
model_name_or_path=/Data/output/
output_dir=/Data/output
question_file=/EyeFM/llava/eval/table/eval.jsonl
image_folder=/
vision_tower=/EyeFM/transfer_image_encoder
accumulation_steps=8
mm_vision_tower=/EyeFM/transfer_image_encoder
answers_file=/EyeFM/llava/eval/ffa_eval/answers.jsonl

CUDA_VISIBLE_DEVICES=7 python /EyeFM/llava/eval/model_vqa_med.py --model-name ${model_name_or_path} --question-file ${question_file} --image-folder ${image_folder} --answers-file ${answers_file}
