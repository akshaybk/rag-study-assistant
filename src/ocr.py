from io import BytesIO
from pathlib import Path
import shutil

from PIL import Image, ImageOps
import pytesseract


COMMON_TESSERACT_PATHS = (
    Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
    Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
)


class TesseractNotInstalledError(RuntimeError):
    """Raised when the Tesseract OCR executable cannot be found."""


def _find_tesseract_executable():
    """Find the Tesseract executable from PATH or common Windows locations."""
    path_from_env = shutil.which("tesseract")
    if path_from_env:
        return path_from_env

    for path in COMMON_TESSERACT_PATHS:
        if path.is_file():
            return str(path)

    return None


def _configure_tesseract():
    """Configure pytesseract to use an installed local Tesseract executable."""
    executable = _find_tesseract_executable()

    if not executable:
        raise TesseractNotInstalledError(
            "Tesseract OCR is not installed or could not be found. "
            "Install Tesseract OCR for Windows, then restart the application."
        )

    pytesseract.pytesseract.tesseract_cmd = executable
    return executable


def extract_text_from_image(image_bytes):
    """Extract text from an uploaded image using local Tesseract OCR.

    Tesseract itself runs locally; no image or text is sent to an API.
    The executable is detected from PATH or common Windows install locations.
    """
    _configure_tesseract()

    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("L")

    # Improve OCR on typical phone/camera question-paper images.
    image = ImageOps.autocontrast(image)

    text = pytesseract.image_to_string(image, config="--psm 6")
    return text.strip()
