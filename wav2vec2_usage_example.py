"""
Wav2Vec2-Large-XLSR-53 Kannada 使用示例代码（根据模型卡片整理）

本脚本仅作为示意代码，展示如何在真实环境中加载
`amoghsgopadi/wav2vec2-large-xlsr-kn` 并进行推理与评估。
当前项目的 WebUI 不会调用该脚本，也不会在启动时下载任何模型权重。
"""

import torch
import torchaudio
from datasets import load_dataset, load_metric
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re


MODEL_ID = "amoghsgopadi/wav2vec2-large-xlsr-kn"
CHARS_TO_IGNORE_REGEX = r"[\,\?\.\!\-\;\\:\\"\\“\%\\‘\\”\\�\\–\\…]"


def load_kannada_dataset(split: str = "test[:10%]"):
    """
    加载基于 OpenSLR Kannada 的数据集切分。

    真实工程中应根据公开示例或自定义数据集格式
    正确填充 `load_dataset` 所需的参数。
    """
    dataset = load_dataset("openslr", "openslr_79", split=split)
    return dataset


def prepare_model_and_processor(device: str = "cuda"):
    """加载 Processor 与声学模型。"""
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    if torch.cuda.is_available() and device == "cuda":
        model.to("cuda")
    return processor, model


def preprocess_batch(batch, resampler, processor):
    """对单条样本进行文本清洗与重采样。"""
    batch["sentence"] = re.sub(CHARS_TO_IGNORE_REGEX, "", batch["sentence"]).lower()
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    batch["speech"] = resampler(speech_array).squeeze().numpy()
    return batch


def evaluate_wer():
    """
    按照模型卡片中的思路，在 10% Kannada 数据上计算 WER。

    注意：本函数不会在 WebUI 演示环境中被调用，
    仅在读者主动执行本脚本时才会触发实际推理。
    """
    test_dataset = load_kannada_dataset()
    wer_metric = load_metric("wer")

    processor, model = prepare_model_and_processor()
    resampler = torchaudio.transforms.Resample(48_000, 16_000)

    test_dataset = test_dataset.map(
        lambda x: preprocess_batch(x, resampler, processor)
    )

    def _evaluate(batch):
        inputs = processor(
            batch["speech"],
            sampling_rate=16_000,
            return_tensors="pt",
            padding=True,
        )
        device = model.device
        with torch.no_grad():
            logits = model(
                inputs.input_values.to(device),
                attention_mask=inputs.attention_mask.to(device),
            ).logits
        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_strings"] = processor.batch_decode(pred_ids)
        return batch

    result = test_dataset.map(_evaluate, batched=True, batch_size=8)
    wer = wer_metric.compute(
        predictions=result["pred_strings"],
        references=result["sentence"],
    )
    print(f"Test WER on OpenSLR kn: {wer * 100:.2f}%")


if __name__ == "__main__":
    # 出于节省资源与网络访问控制的考虑，请在具备合适算力、
    # 并允许下载模型权重的环境下再运行本脚本。
    evaluate_wer()
