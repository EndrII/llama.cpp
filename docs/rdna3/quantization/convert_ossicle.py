"""Convert the Ossicle-4B BF16 checkpoint to GGUF with llama.cpp's own converter.

Ossicle stores the Qwen3.5-4B language model as `llm.*`, the Qwen3-Omni audio encoder as `encoder.*` and the
projection head as `head.*`. Nothing is copied or rewritten on disk: two staging directories hold symlinks to the
safetensors shards plus a config for each half, and the classes below rename tensors on the fly so that the
stock Qwen3.5 (text) and Qwen3-ASR/Omni (audio mmproj) converters can be reused unchanged.

    python convert_ossicle.py stage --src src --lm stage_lm --mm stage_mm
    python convert_ossicle.py -- stage_lm --no-mtp --outtype bf16 --outfile ossicle-4b-bf16.gguf
    python convert_ossicle.py -- stage_mm --mmproj --outtype f16 --outfile mmproj-ossicle-4b-f16.gguf

The stereo adapter (`stereo.*`) has no llama.cpp counterpart and is dropped: GGUF inference is mono.
"""
import json, os, sys
from pathlib import Path

LLAMA = Path(os.environ.get("LLAMA_CPP", "/mnt/data/projects/llama.cpp"))


THINK_OLD = """    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- else %}
        {{- '<think>\\n' }}
    {%- endif %}"""
THINK_NEW = """    {{- '<think>\\n\\n</think>\\n\\n' }}"""


def default_to_non_thinking(template):
    """Ossicle answers directly: it was trained on `<think>\\n\\n</think>\\n\\n` after the assistant tag, so the template always
    emits it, whatever enable_thinking / --reasoning the frontend passes."""
    assert template.count(THINK_OLD) == 1
    return template.replace(THINK_OLD, THINK_NEW)


def stage(src, lm, mm):
    src, lm, mm = Path(src).resolve(), Path(lm), Path(mm)
    cfg = json.load(open(src / "config.json"))
    shared = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "preprocessor_config.json"]
    for d in (lm, mm):
        d.mkdir(parents=True, exist_ok=True)
        for f in ["model.safetensors.index.json"] + sorted(p.name for p in src.glob("model-*.safetensors")) + shared:
            link = d / f
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src / f)
    tc = json.load(open(src / "tokenizer_config.json"))
    tc["chat_template"] = default_to_non_thinking(tc["chat_template"])
    (lm / "tokenizer_config.json").unlink()
    json.dump(tc, open(lm / "tokenizer_config.json", "w"), indent=2)
    (lm / "chat_template.jinja").unlink()
    (lm / "chat_template.jinja").write_text(tc["chat_template"])
    json.dump({**cfg["llm_cfg"], "architectures": ["OssicleLM"], "model_type": "qwen3_5_text"}, open(lm / "config.json", "w"), indent=2)
    audio = {**cfg["audio_cfg"], "output_dim": cfg["llm_cfg"]["hidden_size"]}   # the head projects 1280 -> 2560, not 2048
    json.dump({"architectures": ["OssicleAudio"], "model_type": "qwen3_asr",
               "thinker_config": {"audio_config": audio, "text_config": cfg["llm_cfg"]}}, open(mm / "config.json", "w"), indent=2)


def main():
    sys.path.insert(0, str(LLAMA))
    sys.path.insert(0, str(LLAMA / "gguf-py"))
    import convert_hf_to_gguf as C
    import gguf
    from conversion.base import ModelBase
    from conversion.qwen import Qwen3_5TextModel
    from conversion.qwen3vl import Qwen3ASRMmprojModel

    @ModelBase.register("OssicleLM")
    class OssicleLM(Qwen3_5TextModel):
        model_arch = gguf.MODEL_ARCH.QWEN35

        @classmethod
        def filter_tensors(cls, item):
            name, gen = item
            if not name.startswith("llm."):
                return None
            return super().filter_tensors(("model." + name[len("llm."):], gen))

    @ModelBase.register("OssicleAudio")
    class OssicleAudio(Qwen3ASRMmprojModel):
        def set_gguf_parameters(self):
            super().set_gguf_parameters()
            # Ossicle was trained without <|audio_start|>/<|audio_end|> around the audio embeddings
            self.gguf_writer.add_audio_no_boundary_tokens(True)
            self.gguf_writer.add_audio_exact_frame_count(True)     # ceil(n/hop) mel frames, as WhisperFeatureExtractor
            self.gguf_writer.add_audio_polyphase_resample(True)    # scipy resample_poly-style input resampling

        @classmethod
        def filter_tensors(cls, item):
            name, gen = item
            if name.startswith("encoder."):
                name = "audio_tower." + name[len("encoder."):]
            elif name.startswith("head.proj"):
                name = "audio_tower." + name[len("head."):]
            else:
                return None
            return super().filter_tensors((name, gen))

    import conversion
    conversion.TEXT_MODEL_MAP["OssicleLM"] = "qwen"
    conversion.MMPROJ_MODEL_MAP["OssicleAudio"] = "qwen3vl"
    C.main()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "stage":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd"); ap.add_argument("--src", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--mm", required=True)
        a = ap.parse_args()
        stage(a.src, a.lm, a.mm)
    else:
        sys.argv = [sys.argv[0]] + [x for x in sys.argv[1:] if x != "--"]
        main()
