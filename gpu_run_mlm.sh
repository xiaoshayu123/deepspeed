rm -rf ./test-roberta-base-zero2-multigpu

export BS=32
export NCCL_DEBUG=INFO
export NCCL_SHM_DISABLE=1


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


#rm -rf ./test-roberta-base-zero2-multigpu：删除名为test-roberta-base-zero2-multigpu的目录，如果存在的话。
#export BS=32：设置环境变量BS的值为32，表示批处理大小。
#export NCCL_DEBUG=INFO：设置环境变量NCCL_DEBUG的值为INFO，用于调试NCCL（NVIDIA的集群通信库）。
#export NCCL_SHM_DISABLE=1：设置环境变量NCCL_SHM_DISABLE的值为1，禁用NCCL的共享内存传输。
#deepspeed run_mlm.py：使用DeepSpeed运行名为run_mlm.py的Python脚本，DeepSpeed是一个深度学习优化库。
#--seed 42：设置随机种子为42，用于确保实验可重复。
#--model_name_or_path roberta-base：指定预训练模型的名称或路径，这里使用的是roberta-base模型。
#--dataset_name wikitext，--dataset_config_name wikitext-2-raw-v1：指定数据集名称和配置名称。
#--num_train_epochs 20：设置训练的轮数（epoch）为20。
#--do_train，--do_eval：表示需要进行训练和评估。
#--evaluation_strategy epoch：设置评估策略为每轮训练后进行一次评估。
#--output_dir ./test-roberta-base-zero2-multigpu：设置输出目录。
#--output_figure ./tmp/test-mlm-deepspeed：设置输出图像的路径。
#--out_excel deepspeedMlmMatrix，--out_pic deepspeedTranLosAndAcc：设置输出excel文件和图片的名称。
#--fp16：使用16位浮点数进行训练，可以减少内存消耗并提高训练速度。
#--logging_first_step：从第一步开始记录日志。
#--max_seq_length 300：设置序列的最大长度为300。
#--deepspeed ./ds_zero2_1gpu.json：使用DeepSpeed的配置文件ds_zero2_1gpu.json。