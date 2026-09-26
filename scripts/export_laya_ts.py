"""Export the Laya multilingual checkpoint to browser assets for laya-mbti.

Adapted from `laya-ts/scripts/export_onnx.py` (NandhaKishorM/laya, Apache-2.0).
Changes:
  - the encoder is exported in fp16 (weights + activations) with an fp32
    `last_hidden_state` output, so ONNX Runtime Web's WebGPU provider can run it
    while the head stays fp32;
  - the head stays fp32 and runs on WASM;
  - a `parity.json` fixture with fp32 reference probabilities is written for a
    browser-side numeric check.

Run:
    uv run python scripts/export_laya_ts.py --out-dir public/models/laya
"""

import argparse
import json
import os
import shutil
import sys

DEFAULT_REPO = "convaiinnovations/laya"
DEFAULT_SUBFOLDER = "multilingual"

# Must match src/lib/mbti.ts (concise axis definitions).
AXES = {
    "EI": ("外向 E / 内向 I", {"E": "社交的で人との交流から活力を得る。", "I": "内省的で一人の時間を好む。"}),
    "SN": ("感覚 S / 直観 N", {"S": "具体的・現実的で事実や詳細を重視する。", "N": "抽象的・未来的で可能性や意味を重視する。"}),
    "TF": ("思考 T / 感情 F", {"T": "論理と客観性で判断する。", "F": "共感と調和で判断する。"}),
    "JP": ("判断 J / 知覚 P", {"J": "計画・決定・秩序を好む。", "P": "柔軟・即興・開放性を好む。"}),
}

PARITY_TEXTS = [
    "Just got back from an amazing meetup with 30 new people! Networking events give me so much energy.",
    "Spent the whole weekend alone in my room reading and thinking. The most restorative two days in months.",
    "The argument in that article is logically inconsistent. The data doesn't support the conclusion.",
    "I'm so proud of my friend for getting through this hard time. Being kind matters more than being right.",
]


def download(repo, subfolder, token=None):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    from huggingface_hub import snapshot_download

    prefix = f"{subfolder}/" if subfolder else ""
    allow = [
        prefix + name
        for name in (
            "rl_agent_config.json",
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer/*",
            "encoder/*",
        )
    ]
    root = snapshot_download(repo, allow_patterns=allow, token=token or os.environ.get("HF_TOKEN"))
    path = os.path.join(root, subfolder) if subfolder else root
    if not os.path.isdir(path):
        raise SystemExit(f"subfolder {subfolder!r} not found in {repo!r}")
    return path


def build_questions():
    questions = {}
    for axis, (name, criteria) in AXES.items():
        questions[axis] = {
            "type": "choice",
            "instructions": f"次のX(Twitter)の投稿の書き手の性格傾向を選んでください。観点: {name}",
            "criteria": criteria,
        }
    return questions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--subfolder", default=DEFAULT_SUBFOLDER)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--token", default=None)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file
    from laya.common import build_model

    model_dir = args.model_dir or download(args.repo, args.subfolder, args.token)
    with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    enc_dir = os.path.join(model_dir, "encoder")
    model = build_model(cfg, encoder_dir=enc_dir if os.path.exists(enc_dir) else None)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=False)
    model.eval().float()

    class EncoderOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.encoder = m.encoder

        def forward(self, input_ids, attention_mask):
            # fp16 internals, fp32 output for the head / provider handoff.
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.float()

    class HeadOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.head = m.head
            self.type_emb = m.type_emb
            self.scorer = m.scorer
            self.act_head = m.act_head

        def forward(self, hidden_states, marker_pos, marker_mask, qtype, attention_mask):
            h = hidden_states + self.type_emb(qtype.squeeze(-1))[:, None, :]
            if self.head is not None:
                pad = attention_mask == 0
                for layer in self.head.layers:
                    h = layer(h, src_key_padding_mask=pad)
            idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
            m = torch.gather(h, 1, idx)
            logits = self.scorer(m).squeeze(-1).float()
            logits = logits.masked_fill(~marker_mask, -1e4)
            # Only the option logits are needed. The act/escalate head (softmax, topk,
            # entropy) is unused by this app and blocks the WebGPU EP (TopK), so it is
            # dropped from the exported graph.
            return logits

    def reference_hidden(batch, seq):
        input_ids = torch.ones(batch, seq, dtype=torch.long)
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        with torch.inference_mode():
            return model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    seq = 16
    ref_hidden = reference_hidden(2, seq)
    marker_pos = torch.tensor([[1, 2]] * 2, dtype=torch.long)
    marker_mask = torch.tensor([[True, True]] * 2)
    qtype = torch.tensor([[0]] * 2, dtype=torch.long)
    att = torch.ones(2, seq, dtype=torch.long)

    os.makedirs(args.out_dir, exist_ok=True)
    enc_path = os.path.join(args.out_dir, "encoder.onnx")
    head_path = os.path.join(args.out_dir, "head.onnx")

    batch_dim = torch.export.Dim("batch", min=1, max=128)
    seq_dim = torch.export.Dim("seq", min=1, max=8192)
    markers_dim = torch.export.Dim("markers", min=1, max=256)

    # Head first, while the encoder is still fp32 (ref_hidden is fp32).
    # The dynamo exporter preserves fp16 weights; the legacy TorchScript exporter
    # silently upcasts Linear weights to fp32.
    torch.onnx.export(
        HeadOnly(model).eval(),
        (ref_hidden, marker_pos, marker_mask, qtype, att),
        head_path,
        input_names=["hidden_states", "marker_pos", "marker_mask", "qtype", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "hidden_states": {0: "batch", 1: "seq"},
            "marker_pos": {0: "batch", 1: "markers"},
            "marker_mask": {0: "batch", 1: "markers"},
            "qtype": {0: "batch"},
            "attention_mask": {0: "batch", 1: "seq"},
        },
        dynamic_shapes=(
            {0: batch_dim, 1: seq_dim},
            {0: batch_dim, 1: markers_dim},
            {0: batch_dim, 1: markers_dim},
            {0: batch_dim},
            {0: batch_dim, 1: seq_dim},
        ),
        opset_version=args.opset,
        external_data=False,
    )

    # Reference probabilities (fp32) for the parity fixture, via the Python package.
    import laya

    questions = build_questions()
    parity = {"questions": questions, "texts": PARITY_TEXTS, "expected": []}
    agent = laya.Agent(model_dir)
    try:
        for text in PARITY_TEXTS:
            result = agent.predict(text, questions)
            parity["expected"].append(
                {qid: result["answers"][qid]["probabilities"] for qid in questions}
            )
    finally:
        del agent

    # Encoder fp16 last.
    model.encoder.half()
    input_ids = torch.ones(2, seq, dtype=torch.long)
    attention_mask = torch.ones(2, seq, dtype=torch.long)
    torch.onnx.export(
        EncoderOnly(model).eval(),
        (input_ids, attention_mask),
        enc_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "last_hidden_state": {0: "batch", 1: "seq"},
        },
        dynamic_shapes=({0: batch_dim, 1: seq_dim}, {0: batch_dim, 1: seq_dim}),
        opset_version=args.opset,
        external_data=False,
    )

    for name in ("tokenizer.json", "rl_agent_config.json"):
        src = os.path.join(model_dir, name)
        if not os.path.exists(src) and name == "tokenizer.json":
            src = os.path.join(model_dir, "tokenizer", "tokenizer.json")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(args.out_dir, name))
    with open(os.path.join(args.out_dir, "parity.json"), "w", encoding="utf-8") as f:
        json.dump(parity, f, ensure_ascii=False, indent=2)

    print(f"encoder.onnx: {os.path.getsize(enc_path) / 1e6:.1f} MB")
    print(f"head.onnx:    {os.path.getsize(head_path) / 1e6:.1f} MB")
    print(f"wrote assets to {args.out_dir}")


if __name__ == "__main__":
    main()
