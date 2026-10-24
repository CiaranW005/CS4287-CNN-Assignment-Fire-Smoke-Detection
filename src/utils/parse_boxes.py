from src.config.types import BoundingBox

def parse_boxes(boxes: list[dict]) -> list[BoundingBox]:
    return [
        BoundingBox(
            category=box["category"],
            coordinates=tuple(box["coordinates"])
        )
        for box in boxes
    ]