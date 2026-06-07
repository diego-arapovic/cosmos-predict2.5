"""List the instruction strings + episode counts in the converted YAMS dataset.

Use this to pick a task subset to train on: copy an instruction string verbatim into
W2A_INSTRUCTIONS="..." (pipe-separate multiple, exact match) for train.sh. Reads the `instruction`
zarr attr that `process_recordings.py` wrote -- the same attr `MimicDataset(instruction_filter=...)`
filters on, so the strings are guaranteed to match.

    python mimic_video_port/commands/list_instructions.py [CONVERTED_DIR]

CONVERTED_DIR defaults to $W2A_CONVERTED, then /opt/dlami/nvme/world2action/teleop_converted.
"""
import collections
import os
import pathlib
import sys

import zarr


def main() -> None:
    default = os.environ.get("W2A_CONVERTED", "/opt/dlami/nvme/world2action/teleop_converted")
    data_dir = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else default)
    paths = sorted(data_dir.glob("**/*.zarr"))

    counts: collections.Counter[str] = collections.Counter()
    for p in paths:
        try:
            counts[str(zarr.open(str(p), "r").attrs.get("instruction", "<no instruction attr>"))] += 1
        except Exception as e:  # noqa: BLE001 - report unreadable episodes instead of crashing
            counts[f"<error: {e}>"] += 1

    print(f"{len(paths)} episodes under {data_dir}")
    print(f"{len(counts)} unique instructions (count  instruction):\n")
    for instruction, n in counts.most_common():
        print(f"  {n:5d}  {instruction!r}")

    if counts:
        top = counts.most_common(1)[0][0]
        print(
            "\nTrain just this task, e.g.:\n"
            f'  W2A_INSTRUCTIONS="{top}" EXP=yams_medium bash mimic_video_port/commands/train.sh'
        )


if __name__ == "__main__":
    main()
