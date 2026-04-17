from pathlib import Path

import click
import cv2
from tqdm import tqdm

from pipeline import pipeline

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")


@click.command()
@click.argument("input_dir")
@click.argument("output_dir")
def deteriorate(input_dir: str, output_dir: str) -> None:
    """Make a worsened copy of each image from INPUT_DIR by applying a random set
    of transformations. The results are then put into OUTPUT_DIR under their
    respective filenames."""

    input_path: Path = Path(input_dir)
    output_path: Path = Path(output_dir)

    if not input_path.exists():
        raise click.BadParameter(f"invalid input directory `{input_dir}`")

    for img_path in tqdm(input_path.rglob("*"), desc="Processing images", unit="img"):
        if img_path.suffix in SUPPORTED_EXTENSIONS:
            image = cv2.imread(img_path)

            if image is None:
                click.echo(f"Image {img_path} could not be read by cv2. Skipping...")
                continue

            new_image_path = output_path / Path(*img_path.parts[1:])
            new_image_path.parent.mkdir(parents=True, exist_ok=True)

            augmented_image = pipeline(image=image)["image"]
            cv2.imwrite(new_image_path, augmented_image)


if __name__ == "__main__":
    deteriorate()
