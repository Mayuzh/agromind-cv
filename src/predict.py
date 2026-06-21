import argparse
import json
from pathlib import Path

from agromind_cv.inference import predict_image


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one tomato leaf image.")
    parser.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(predict_image(args.image), indent=2))


if __name__ == "__main__":
    main()

