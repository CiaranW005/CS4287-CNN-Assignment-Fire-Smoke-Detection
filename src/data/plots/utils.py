def image_category(boxes):
    categories = {box["category"] for box in boxes}

    if categories == {0, 1}:
        return "both"
    elif categories == {1}:
        return "fire_only"
    elif categories == {0}:
        return "smoke_only"
    else:
        return "neither"