"""Precompute & cache Reason1 language embeddings for world2action (the T5 -> Reason1 swap).

Replaces mimic-video's `precompute_t5.py`. Reason1 `full_concat` is 100352-d (~102 MB/episode), so we
build ONE shared cache keyed by instruction string (teleop has few unique instructions) instead of
storing the embedding per-episode in zarr -- the dataset's `Reason1EmbeddingLookup` transform reads
each episode's `language_instruction` and looks it up (mmap'd, shared across dataloader workers).
See DESIGN_RISKS.md #16.

    python mimic_video_port/data_preprocessing/precompute_reason1.py \
        --data-dir data/teleop_converted            # writes <data-dir>/reason1_embeddings.pt

Requires `hf auth login` + access to gated nvidia/Cosmos-Reason1-7B (first run downloads ~15 GB).
"""
import argparse
import pathlib

import torch
import zarr

from mimic_video_port.world2action.checkpoints import REASON1_UUID, register_external_checkpoints

DEFAULT_CACHE_NAME = "reason1_embeddings.pt"


def discover_instructions(data_dir: pathlib.Path) -> dict[str, list[str]]:
    """Map each unique instruction string -> list of episode zarr names that use it."""
    instructions: dict[str, list[str]] = {}
    for zpath in sorted(pathlib.Path(data_dir).glob("*.zarr")):
        root = zarr.open(str(zpath), "r")
        instr = root.attrs.get("instruction")
        if instr is None:  # fall back to the stored language_instruction array
            instr = bytes(root["language_instruction"][0]).decode("utf-8")
        instructions.setdefault(instr.strip(), []).append(zpath.name)
    return instructions


def build_cache(data_dir: pathlib.Path, out_path: pathlib.Path, device: str = "cuda") -> dict[str, torch.Tensor]:
    """Encode unique instructions with Reason1, saving {instruction: (L, 100352) bf16}.

    Incremental: loads the existing cache and only encodes instructions not already present, so adding
    episodes with already-seen instructions is free, and the 7B encoder is not even loaded if nothing is new.
    """
    instructions = discover_instructions(data_dir)
    if not instructions:
        raise SystemExit(f"no *.zarr episodes with an instruction found under {data_dir}")

    cache: dict[str, torch.Tensor] = {}
    if out_path.exists():
        cache = torch.load(out_path, map_location="cpu", weights_only=False)
    new_instrs = [i for i in instructions if i not in cache]
    print(f"[reason1] {len(instructions)} unique instruction(s); {len(new_instrs)} new ({len(cache)} already cached)")
    if not new_instrs:
        print(f"[reason1] cache up to date -> {out_path}")
        return cache

    register_external_checkpoints()
    from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig

    enc = TextEncoder(
        TextEncoderConfig(compute_online=True, embedding_concat_strategy="full_concat", ckpt_path=REASON1_UUID),
        device=device,
    )
    with torch.no_grad():
        for instr in new_instrs:
            emb = enc.compute_text_embeddings_online({"ai_caption": [instr]}, "ai_caption")  # (1, L, 100352)
            # Store in the backbone's compute dtype (bf16 = Reason1's own dtype): matches the net, avoids
            # the fp16-overflow risk, same size. The dataset/backbone also casts defensively at load.
            cache[instr] = emb.squeeze(0).to(torch.bfloat16).cpu().contiguous()  # (L, 100352)
            print(f"  '{instr[:60]}' -> {tuple(cache[instr].shape)}  ({len(instructions[instr])} episode(s))")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(f"[reason1] saved {len(cache)} embedding(s) ({len(new_instrs)} new) -> {out_path}")
    return cache


def main() -> None:
    p = argparse.ArgumentParser(description="Precompute Reason1 instruction embeddings for world2action.")
    p.add_argument("--data-dir", type=pathlib.Path, required=True, help="Dir of converted episode_*.zarr files.")
    p.add_argument("--out", type=pathlib.Path, default=None, help="Output .pt (default: <data-dir>/reason1_embeddings.pt).")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    out = args.out or (args.data_dir / DEFAULT_CACHE_NAME)
    build_cache(args.data_dir, out, device=args.device)


if __name__ == "__main__":
    main()
