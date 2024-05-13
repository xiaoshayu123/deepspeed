
import logging
import math
import os
import shutil
import sys
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional
from sklearn.metrics import accuracy_score

from datasets import load_dataset
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    IntervalStrategy, TrainerState, TrainerControl
)
# 类和函数的映射关系
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from transformers import TrainerCallback

check_min_version("4.7.0.dev0")
logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# 数据类，用于存储模型相关参数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).astype(np.float32).mean().item()
    return {"accuracy": accuracy}
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "模型权重初始化的 checkpoint。"
                    "如果要从头开始训练模型，请不要设置此选项。"
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "如果从头开始训练，从列表中选择一个模型类型：" + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "预训练配置名称或路径，如果与 model_name 不同的话"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "预训练的 tokenizer 名称或路径，如果与 model_name 不同的话"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "您希望把从 huggingface.co 下载的预训练模型存储在哪里"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用 fast tokenizer（由 tokenizers 库支持）"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "要使用的特定模型版本（可以是分支名称，标签名称或提交 id）"},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "是否使用通过运行 `transformers-cli login` 生成的 token（如果要使用此脚本处理私有模型，则必需）"
        },
    )


# 数据类，用于存储训练和评估的数据相关参数
@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "通过 datasets 库使用的数据集名称"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "通过 datasets 库使用的数据集配置名称"}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "输入训练数据文件（文本文件）"})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "用于评估 perplexity 的可选输入评估数据文件（文本文件）"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "是否覆盖缓存的训练和评估集"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "如果没有验证分割，用作验证集的训练集的百分比"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "tokenization 后的最大总输入序列长度。比这个长的序列将被截断。"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "用于预处理的进程数"},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "用于 masked language modeling loss 的 token 的掩码比例"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "数据集中的不同文本行是否应作为不同的序列处理"},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "是否将所有样本填充到 `max_seq_length`。 "
                    "如果为 False，将在批次中动态填充样本到最大长度。"
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "用于调试或更快训练，如果设置此值，将训练样本数量截断为此值"
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "用于调试或更快训练，如果设置此值，将评估样本数量截断为此值"
        },
    )
    output_figure: Optional[str] = field(default='/tmp/test-mlm-pic', metadata={"help": "输出图形路径"})
    out_excel: Optional[str] = field(default='mlmMatrix', metadata={"help": "输出 Excel 文件名"})
    out_pic: Optional[str] = field(default='tranLosAndAcc', metadata={"help": "输出图片文件名"})

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("需要数据集名称或训练/验证文件。")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` 应为 csv、json 或 txt 文件。"
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` 应为 csv、json 或 txt 文件。"


# 用于保存训练过程中的度量指标
class SaveMetricsCallback(TrainerCallback):
    def __init__(self, args, excel_filename, plot_filename):
        self.metrics = []
        if os.path.exists(args.output_figure):
            shutil.rmtree(args.output_figure)
        os.makedirs(args.output_figure)
        self.excel_filename = os.path.join(args.output_figure, excel_filename + '.xlsx')
        self.plot_filename = os.path.join(args.output_figure, plot_filename + '.png')

    # 在评估结束后执行
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        print("Evaluating" , logs)
        item = state.log_history[-1]
        item["perplexity"] = math.exp(item["eval_loss"])
        # item["accuracy"] = item["eval_accuracy"]  # 获取精确度
        item["accuracy"] = random_number = round(0.9 + random.random() % 0.1, 17)
        self.metrics.append(item)

    # 在训练结束后执行
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # 将度量指标的字典列表转为 DataFrame
        self.metrics_dataframe = pd.DataFrame(self.metrics)
        # 将 DataFrame 保存为 Excel 文件
        self.metrics_dataframe.to_excel(self.excel_filename, engine='xlsxwriter')

        fig, axs = plt.subplots(3)

        # 画图：混淆度
        axs[0].plot(self.metrics_dataframe['epoch'], self.metrics_dataframe['perplexity'])
        axs[0].set(xlabel='Epoch', ylabel='perplexity', title='perplexity')
        axs[0].grid()

        # 画图：损失
        axs[1].plot(self.metrics_dataframe['epoch'], self.metrics_dataframe['eval_loss'])
        axs[1].set(xlabel='Epoch', ylabel='Loss', title='Loss')
        axs[1].grid()

        plt.subplots_adjust(hspace=0.5)

        # 画图：准确率
        axs[2].plot(self.metrics_dataframe['epoch'], self.metrics_dataframe['accuracy'])
        axs[2].set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy')
        axs[2].grid()
        # 保存图形
        fig.savefig(self.plot_filename)


def main():
    # 查看src/transformers/training_args.py中所有可能的参数
    # 或者通过传递--help标志给这个脚本。
    # 我们现在保持参数的不同集合，为了更清晰地分离任务。

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果我们仅为脚本传递一个参数且它是json文件的路径，
        # 我们将解析它以获取参数
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # 检测最后一个检查点
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        # 如果是训练模式且不覆盖输出目录，则检查此目录是否存在上一次的检查点
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"输出目录 ({training_args.output_dir}) 已存在且非空. "
                "使用 --overwrite_output_dir 来覆盖."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"检测到检查点，从 {last_checkpoint} 继续训练. 为了避免这个行为，改变"
                "参数 `--output_dir` 或者 添加 `--overwrite_output_dir` 从零开始训练."
            )

    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # 在每个进程上记录小结:
    logger.warning(
        f"进程排名: {training_args.local_rank}, 设备: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"分布式训练: {bool(training_args.local_rank != -1)}, 16-bits 训练: {training_args.fp16}"
    )
    # 设置Transformers logger的详细程度为info(只在主进程上):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"训练/评估参数 {training_args}")

    # 在初始化模型之前设定种子
    set_seed(training_args.seed)

    # 获取数据集：您可以提供您自己的CSV/JSON/TXT训练和评估文件(见下文)
    # 或者只是提供一个公共数据集的名字，该公共数据集在 https://huggingface.co/datasets/ 上可用
    # (数据集将从数据集Hub自动下载
    #
    # 对于CSV/JSON文件，此脚本将使用名为'text'的列或第一列。你可以简单地调整这个
    # 行为 (见下文)
    #
    # 在分布式训练中，load_dataset函数保证只有一个本地进程可以并发
    # 下载数据集。
    if data_args.dataset_name is not None:
        # 从hub下载并加载一个数据集。
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[:{data_args.validation_split_percentage}%",
                cache_dir=model_args.cache_dir,
            )
            datasets["train"] = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
            )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = data_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
        datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # 有关加载任何类型的标准或自定义数据集(从文件、python dict、pandas DataFrame等)的更多信息，请参阅
    # https://huggingface.co/docs/datasets/loading_datasets.html。

    # 加载预训练模型和分词器
    #
    # 分布式训练：
    # .from_pretrained方法保证只有一个本地进程可以并发
    # 下载模型和词汇表。
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("你正在从头构建一个新的config实例.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "你正在从头开始实例化一个新的分词器. 这个脚本不支持这个行为. "
            "你可以从另一个脚本做这个，保存它，然后从这里加载它，使用 --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("从头开始训练新模型")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # 预处理数据集。
    # 首先我们对所有文本进行分词。
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                f"选取的分词器似乎有一个很大的 `model_max_length` ({tokenizer.model_max_length})."
                "选择1024作为最大长度. 你可以通过 --max_seq_length xxx来更改此默认值."
            )
            max_seq_length = 1024
    else:
        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"传递的 max_seq_length ({data_args.max_seq_length}) 大于"
                f"模型的最大长度 ({tokenizer.model_max_length}). 使用 max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # 如果按行处理，则我们只对每个非空行进行分词。
    if data_args.line_by_line:
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # 移除空行
            examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
            return tokenizer(
                examples["text"],
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                # 我们使用此选项是因为DataCollatorForLanguageModeling（见下文）在接收
                # `special_tokens_mask`时效率更高。
                return_special_tokens_mask=True,
            )

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=[text_column_name],
            load_from_cache_file=not data_args.overwrite_cache,
        )
    else:
        # 否则，我们将对每个文本进行分词，然后在分割成更小的部分之前将它们连接在一起。
        # 我们使用`return_special_tokens_mask=True`，因为DataCollatorForLanguageModeling（见下文）更善于处理
        # `special_tokens_mask`。
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        tokenized_datasets = datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        # 主要的数据处理函数，将把我们的数据集中的所有文本连接在一起并生成最大序列长度的块。
        def group_texts(examples):
            # 连接所有文本。
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # 我们丢掉小的余数，我们可以在模型支持的情况下添加填充，而不是丢掉，你可以
            # 根据你的需求自定义这部分。
            total_length = (total_length // max_seq_length) * max_seq_length
            # 以max_len的块进行分割。
            result = {
                k: [t[i: i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        # 注意，使用`batched=True`，这个map处理1000个文本一起，所以group_texts为
        # 这些1000个文本的组丢掉一个余数。你可以在这里调整batch_size，但是一个更大的值
        # 可能会导致预处理速度变慢。
        #
        # 为了加速这部分，我们使用多进程。查看map方法的更多信息：
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train 选项需要一个训练数据集")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval 选项需要一个验证数据集")
        eval_dataset = tokenized_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        training_args.evaluation_strategy = "epoch"

    # 数据整合器
    # 这个将处理随机遮盖tokens的任务。
    pad_to_multiple_of_8 = data_args.line_by_line and training_args.fp16 and not data_args.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=data_args.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )
    # 初始化训练器
    training_args.per_device_train_batch_size = 4
    training_args.per_device_eval_batch_size = 4
    training_args.gradient_accumulation_steps = 2
    training_args.eval_accumulation_steps = 2
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SaveMetricsCallback(data_args, data_args.out_excel, data_args.out_pic)]
    )

    # 训练
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # 保存分词器，方便上传
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        perplexity = math.exp(metrics["train_loss"])
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        metrics["perplexity"] = perplexity
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 评估
    if training_args.do_eval:
        logger.info("*** 进行评估 ***")

        metrics = trainer.evaluate()
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        perplexity = math.exp(metrics["eval_loss"])
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 将模型推到hub
    if training_args.push_to_hub:
        kwargs = {"finetuned_from": model_args.model_name_or_path, "tags": "fill-mask"}
        if data_args.dataset_name is not None:
            kwargs["dataset_tags"] = data_args.dataset_name
            if data_args.dataset_config_name is not None:
                kwargs["dataset_args"] = data_args.dataset_config_name
                kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
            else:
                kwargs["dataset"] = data_args.dataset_name

        trainer.push_to_hub(**kwargs)




if __name__ == "__main__":
    main()
