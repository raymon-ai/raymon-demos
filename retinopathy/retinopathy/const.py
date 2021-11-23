from pathlib import Path
import os

TAG_CHOICES = {
    "age": range(30, 90),
    "hospital": ["St Jozef", "Maria Middelares", "UZA", "UZ Gent"],
    "eye": ["left", "right"],
    "machine_id": {
        "St Jozef": [
            "a466a0ff-da21-4522-816e-08f89bd213b4",
            "99d332c8-7064-44a8-a649-6c767bdb0ce9",
        ],
        "Maria Middelares": [
            "de576b99-6aea-4802-990f-c34b1cecb248",
            "fcb0d900-1a80-454f-84fc-7f4bdd1a3fbf",
        ],
        "UZA": [
            "2f930ab2-3e8a-4869-a157-1bc5cd327244",
            "c098be7e-b09a-4584-ad7d-13b505e4b0f3",
        ],
        "UZ Gent": [
            "1dbdc031-d805-4ffe-9800-e53aaddd02a7",
            "9e2e7870-1de4-40b6-914a-8b4ebc7edc07",
        ],
    },
}
BAD_MACHINE = TAG_CHOICES["machine_id"]["UZA"][0]
ROOT = Path("..").resolve()
SECRET = Path(
    os.environ.get("RAYMON_CLIENT_SECRET_FILE", ROOT / "m2mcreds-retinopathy.json")
)
