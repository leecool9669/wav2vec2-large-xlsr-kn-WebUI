import gradio as gr

# 注意：本示例仅为界面展示，不实际下载或加载 wav2vec2-large-xlsr-kn 模型权重。
# 在真实部署场景中，可在此处接入 HuggingFace Transformers 与本地或远程模型服务。

def dummy_asr(audio, language, sample_rate):
    """占位推理函数：模拟 Kannada 语音识别流程，不进行真实推理。"""
    if audio is None:
        return "请先上传一段包含 Kannada 语音的音频片段。"
    duration = getattr(audio, "duration", None) or 0.0
    duration_str = f"约 {duration:.1f} 秒" if duration > 0 else "未知时长"
    return (
        "【占位识别结果】\n"
        "本演示版本不会下载或加载任何真实模型权重，仅用于展示 wav2vec2-large-xlsr-kn 在界面层面的典型使用流程。\n\n"
        f"· 语言设置：{language}（示例中对应 Kannada）\n"
        f"· 采样率：{sample_rate} Hz（示例中推荐 16000 Hz）\n"
        f"· 音频时长：{duration_str}\n\n"
        "在实际系统中，该区域将展示模型输出的转写文本，并可附带词错误率 (WER)、置信度统计以及与参考文本的对齐结果。"
    )

with gr.Blocks(title="Wav2Vec2-Large-XLSR-53 Kannada WebUI") as demo:
    gr.Markdown(
        """# Wav2Vec2-Large-XLSR-53 Kannada WebUI 演示界面

本页面用于演示基于 `wav2vec2-large-xlsr-kn` 预训练语音识别模型构建的交互式 WebUI 结构。
当前实现不下载模型权重，也不执行真实推理，仅通过占位输出展示从音频输入到文本结果呈现的完整交互链路。"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. 输入配置与音频上传")
            language = gr.Dropdown(
                label="识别语言 (Language)",
                choices=["Kannada (kn)", "English", "Multilingual"],
                value="Kannada (kn)",
            )
            sample_rate = gr.Slider(
                label="采样率 (Hz)", minimum=8000, maximum=48000, step=1000, value=16000
            )
            audio_in = gr.Audio(
                label="上传或录制待识别音频 (单声道)",
                type="filepath",
                sources=["upload", "microphone"],
            )
            run_btn = gr.Button("执行占位识别 (不实际调用模型)")

        with gr.Column(scale=1):
            gr.Markdown("### 2. 文本结果与分析可视化")
            text_out = gr.Textbox(
                label="占位转写结果",
                lines=10,
                interactive=False,
            )

    def _wrapper(audio_path, lang, sr):
        if audio_path is None:
            return "请先在左侧上传或录制音频。"
        class _DummyAudio:
            def __init__(self, path):
                self.path = path
                self.duration = 0.0
        audio = _DummyAudio(audio_path)
        return dummy_asr(audio, lang, int(sr))

    run_btn.click(_wrapper, inputs=[audio_in, language, sample_rate], outputs=[text_out])

if __name__ == "__main__":
    # 使用 7862 端口以避免与其他演示 WebUI 冲突
    demo.launch(server_name="0.0.0.0", server_port=7862, debug=False)
