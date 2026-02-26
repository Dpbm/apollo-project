import os

IMAGES_PATH = os.path.join(".", "assets")
BANNER_IMAGE = os.path.join(IMAGES_PATH, "banner.png")
MAX_JOBS = 1 if os.cpu_count() < 4 else 4
RANDOM_STATE = 42