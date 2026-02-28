"""
COCO Dataset Classes and Utilities.

COCO (Common Objects in Context) is a large-scale object detection dataset.
It contains 80 object categories that are commonly found in everyday scenes.

This module provides:
- COCO_CLASSES: List of all 80 class names
- COCO_COLORS: Pre-generated colors for each class
- Helper functions for class name/color lookup

Example:
    >>> from src.constants import COCO_CLASSES, get_class_name
    >>> COCO_CLASSES[0]
    'person'
    >>> get_class_name(2)
    'car'
"""

import colorsys

# =============================================================================
# COCO CLASSES (80 categories)
# =============================================================================
# Source: https://cocodataset.org/#explore
# Format: class_id -> class_name
# Note: IDs are 0-indexed (0 = person, 1 = bicycle, etc.)

COCO_CLASSES: list[str] = [
    "person",  # 0
    "bicycle",  # 1
    "car",  # 2
    "motorcycle",  # 3
    "airplane",  # 4
    "bus",  # 5
    "train",  # 6
    "truck",  # 7
    "boat",  # 8
    "traffic light",  # 9
    "fire hydrant",  # 10
    "stop sign",  # 11
    "parking meter",  # 12
    "bench",  # 13
    "bird",  # 14
    "cat",  # 15
    "dog",  # 16
    "horse",  # 17
    "sheep",  # 18
    "cow",  # 19
    "elephant",  # 20
    "bear",  # 21
    "zebra",  # 22
    "giraffe",  # 23
    "backpack",  # 24
    "umbrella",  # 25
    "handbag",  # 26
    "tie",  # 27
    "suitcase",  # 28
    "frisbee",  # 29
    "skis",  # 30
    "snowboard",  # 31
    "sports ball",  # 32
    "kite",  # 33
    "baseball bat",  # 34
    "baseball glove",  # 35
    "skateboard",  # 36
    "surfboard",  # 37
    "tennis racket",  # 38
    "bottle",  # 39
    "wine glass",  # 40
    "cup",  # 41
    "fork",  # 42
    "knife",  # 43
    "spoon",  # 44
    "bowl",  # 45
    "banana",  # 46
    "apple",  # 47
    "sandwich",  # 48
    "orange",  # 49
    "broccoli",  # 50
    "carrot",  # 51
    "hot dog",  # 52
    "pizza",  # 53
    "donut",  # 54
    "cake",  # 55
    "chair",  # 56
    "couch",  # 57
    "potted plant",  # 58
    "bed",  # 59
    "dining table",  # 60
    "toilet",  # 61
    "tv",  # 62
    "laptop",  # 63
    "mouse",  # 64
    "remote",  # 65
    "keyboard",  # 66
    "cell phone",  # 67
    "microwave",  # 68
    "oven",  # 69
    "toaster",  # 70
    "sink",  # 71
    "refrigerator",  # 72
    "book",  # 73
    "clock",  # 74
    "vase",  # 75
    "scissors",  # 76
    "teddy bear",  # 77
    "hair drier",  # 78
    "toothbrush",  # 79
]

# Number of classes (should be 80)
NUM_CLASSES = len(COCO_CLASSES)


# =============================================================================
# COLOR GENERATION
# =============================================================================
# We generate colors using HSV color space to ensure:
# 1. All colors are visually distinct
# 2. Colors are bright and saturated (good for visualization)
# 3. Consistent colors across runs


def _generate_class_colors(num_classes: int = NUM_CLASSES) -> list[tuple[int, int, int]]:
    """
    Generate distinct colors for each class using HSV color space.

    This spreads hues evenly across the color spectrum, ensuring
    each class has a visually distinct color.

    Args:
        num_classes: Number of colors to generate

    Returns:
        List of BGR color tuples (for OpenCV compatibility)
    """
    colors = []
    for i in range(num_classes):
        # Spread hues evenly: 0.0 to 1.0 in steps
        hue = i / num_classes

        # Fixed saturation and value for bright, visible colors
        saturation = 0.9  # High saturation (vivid colors)
        value = 0.9  # High brightness

        # Convert HSV to RGB (returns 0-1 range)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)

        # Convert to BGR (OpenCV format) with 0-255 range
        # Note: OpenCV uses BGR order, not RGB
        bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        colors.append(bgr)

    return colors


# Pre-generate colors for all classes (computed once at import)
COCO_COLORS: list[tuple[int, int, int]] = _generate_class_colors()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_class_name(class_id: int) -> str:
    """
    Get class name from class ID.

    Args:
        class_id: COCO class ID (0-79)

    Returns:
        Class name string, or "unknown" if ID is invalid

    Example:
        >>> get_class_name(0)
        'person'
        >>> get_class_name(16)
        'dog'
        >>> get_class_name(999)
        'unknown'
    """
    if 0 <= class_id < NUM_CLASSES:
        return COCO_CLASSES[class_id]
    return "unknown"


def get_class_color(class_id: int) -> tuple[int, int, int]:
    """
    Get BGR color for a class ID.

    Args:
        class_id: COCO class ID (0-79)

    Returns:
        BGR color tuple for OpenCV drawing

    Example:
        >>> color = get_class_color(0)  # person
        >>> color  # Some color like (144, 238, 144)
    """
    if 0 <= class_id < NUM_CLASSES:
        return COCO_COLORS[class_id]
    # Return a default gray for unknown classes
    return (128, 128, 128)


# =============================================================================
# COMMON CLASS GROUPS (for filtering)
# =============================================================================

# Pre-defined class groups for common use cases
PERSON_CLASSES = [0]  # person only

VEHICLE_CLASSES = [  # Transportation
    1,  # bicycle
    2,  # car
    3,  # motorcycle
    4,  # airplane
    5,  # bus
    6,  # train
    7,  # truck
    8,  # boat
]

ANIMAL_CLASSES = [
    14,  # bird
    15,  # cat
    16,  # dog
    17,  # horse
    18,  # sheep
    19,  # cow
    20,  # elephant
    21,  # bear
    22,  # zebra
    23,  # giraffe
]

# For traffic/surveillance applications
TRAFFIC_CLASSES = [0, 2, 3, 5, 6, 7]  # person, car, motorcycle, bus, train, truck
