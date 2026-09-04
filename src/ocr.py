from io import BytesIO

from PIL import Image, ImageOps
import pytesseract


def extract_text_from_image(image_bytes):
    """Extract text from an uploaded image using local Tesseract OCR.

    Tesseract itself runs locally; no image or text is sent to an API.
    """
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("L")

    # Improve OCR on typical phone/camera question-paper images.
    image = ImageOps.autocontrast(image)

    text = pytesseract.image_to_string(image, config="--psm 6")
    return text.strip()
