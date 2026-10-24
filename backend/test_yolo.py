"""Small smoke test for an Ultralytics YOLO model.

Run with:
	python test_yolo.py --model yolov8n.pt --source path/to/image.jpg
"""

import argparse
from pathlib import Path


def main() -> None:
	parser = argparse.ArgumentParser(description="Check that a YOLO model loads and runs inference.")
	parser.add_argument("--model", default="yolov8n.pt", help="Model weights or model path")
	parser.add_argument("--source", default=None, help="Image/video path (optional)")
	args = parser.parse_args()

	try:
		from ultralytics import YOLO
	except ImportError:
		print("YOLO is not installed. Run: pip install ultralytics")
		return

	if args.source and not Path(args.source).exists():
		print(f"Source not found: {args.source}")
		return

	try:
		model = YOLO(args.model)
		output_dir = Path(__file__).resolve().parent.parent / "results"
		results = model.predict(
			source=args.source or "https://ultralytics.com/images/bus.jpg",
			save=True,
			project=output_dir,
			exist_ok=True,
			verbose=False,
		)
		print(f"YOLO is working: model loaded and inference completed ({len(results)} result(s)).")
		print(f"Saved output to: {output_dir / 'yolo_test'}")
		for result in results:
			print(f"Detections: {len(result.boxes)}")
	except Exception as error:
		print(f"YOLO test failed: {error}")


if __name__ == "__main__":
	main()
