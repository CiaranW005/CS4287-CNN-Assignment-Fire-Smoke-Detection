import pandas as pd
import json

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

from src.config.paths import DATASET_DIR, IMAGE_DIR, RESULTS_DIR

IMAGE_DIR = IMAGE_DIR / "data_analysis"
df = pd.read_csv(RESULTS_DIR / "csvs/data/metadata_raw.csv")
df["boxes"] = df["boxes"].apply(json.loads)

def plot_zero_boxes():
    sizes_df = df["boxes"].explode().dropna().apply(pd.Series)

    sizes_df[["xc", "yc", "width", "height"]] = pd.DataFrame(
        sizes_df["coordinates"].tolist(),
        index=sizes_df.index
    )

    sizes_df["area"] = sizes_df["width"] * sizes_df["height"]
    sizes_df["split"] = df.loc[sizes_df.index, "split"].to_numpy()

    zero_boxes = sizes_df[sizes_df["area"] == 0]

    print("Zero-area boxes:", len(zero_boxes))

    for idx in zero_boxes.index.unique():
        record = df.loc[idx]
        image_path = DATASET_DIR / record["path"]

        with Image.open(image_path) as img:
            image = img.copy()

        width, height = image.size

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(image)

        for box in record["boxes"]:
            xc, yc, w, h = box["coordinates"]

            x = xc * width
            y = yc * height

            category = "Fire" if box["category"] == 1 else "Smoke"
            if w <= 0 or h <= 0:
                # White outline to make the marker visible
                ax.scatter(
                    x, y,
                    c="white",
                    marker="x",
                    s=350,
                    linewidths=6,
                    zorder=10,
                    clip_on=False
                )

                # Red X marking invalid annotation centre
                ax.scatter(
                    x, y,
                    c="red",
                    marker="x",
                    s=350,
                    linewidths=3,
                    zorder=11,
                    clip_on=False
                )

                ax.annotate(
                    category,
                    (x, y),
                    xytext=(12, -15),
                    textcoords="offset points",
                    color="white",
                    fontsize=11,
                    fontweight="bold",
                    bbox=dict(
                        facecolor="red",
                        edgecolor="white",
                        alpha=0.85,
                        pad=3
                    ),
                    zorder=12
                )

            else:
                rect = patches.Rectangle(
                    ((xc - w / 2) * width, (yc - h / 2) * height),
                    w * width,
                    h * height,
                    linewidth=2,
                    edgecolor="red" if category == "Fire" else "gray",
                    facecolor="none"
                )
                ax.add_patch(rect)

        ax.set_title(f"{image_path.name} | Index: {idx}")
        ax.set_xlim(-15, width + 15)
        ax.set_ylim(height + 15, -15)
        ax.axis("off")

        fig.savefig(IMAGE_DIR / f"error_box_{idx}.png", bbox_inches="tight")
        plt.close(fig)
    return list(IMAGE_DIR.glob("error_box_*.png"))