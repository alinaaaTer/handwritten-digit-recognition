# src/io_validators.py

MAX_FILE_SIZE_MB = 5
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def validate_uploaded_file(filename: str, file_bytes: bytes) -> None:
    """
    Validates uploaded file by:
    - extension (png/jpg/jpeg)
    - size (<= 5 MB)

    Raises ValueError with a user-friendly message if validation fails.
    """
    if not filename:
        raise ValueError("Empty filename")

    if not file_bytes:
        raise ValueError("Empty file")

    ext = "." + filename.split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file format. Allowed: PNG, JPG, JPEG")

    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB} MB limit")
