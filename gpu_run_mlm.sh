rm -rf ./test-roberta-base-zero2-multigpu



deepspeed run_mlm.py \
--seed 42 \
--model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
--num_train_epochs 20 \
--do_train \
--do_eval \
--evaluation_strategy epoch  \
--output_dir ./test-roberta-base-zero2-multigpu \
--output_figure ./tmp/test-mlm-deepspeed \
--out_excel deepspeedMlmMatrix \
--out_pic deepspeedTranLosAndAcc \
--fp16 \
--logging_first_step \
--max_seq_length 300 \
--deepspeed ./ds_zero2_1gpu.json  \


