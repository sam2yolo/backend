"""Assemble annotated frames from a SAM video run into an mp4.

Usage:
    python assemble_video.py outputs/video --fps 6 --out outputs/video_annotated.mp4
"""

import argparse
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames_dir", help="directory of annotated *.jpg frames")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=6.0)
    args = ap.parse_args()

    frames_dir = Path(args.frames_dir)
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        raise SystemExit(f"No .jpg frames in {frames_dir}")

    out = Path(args.out) if args.out else frames_dir.parent / "video_annotated.mp4"
    first = cv2.imread(str(frames[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
    )
    for f in frames:
        img = cv2.imread(str(f))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
    writer.release()
    print(f"Wrote {out}  ({len(frames)} frames @ {args.fps} fps)")


if __name__ == "__main__":
    main()
