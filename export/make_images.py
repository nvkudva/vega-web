"""Synthetic document images for the image-path parity set. Sizes are chosen to exercise the
processor's resize: one already on the 32-pixel grid, one off it, one over max_pixels."""
import os
from PIL import Image, ImageDraw, ImageFont

OUT = "../golden/images"


def font(size):
    for p in ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def page(w, h, lines, title=None, box=False):
    im = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(im)
    y = 30
    if title:
        d.text((30, y), title, fill="black", font=font(int(h * 0.06)))
        y += int(h * 0.1)
    for ln in lines:
        d.text((30, y), ln, fill=(40, 40, 40), font=font(int(h * 0.032)))
        y += int(h * 0.05)
    if box:
        d.rectangle([w * 0.55, h * 0.72, w - 30, h - 40], outline="black", width=3)
        d.text((w * 0.57, h * 0.75), "TOTAL DUE  $1,284.50", fill="black", font=font(int(h * 0.035)))
    return im


def main():
    os.makedirs(OUT, exist_ok=True)
    page(640, 480, ["Invoice no. 88123", "Date: 2026-09-14", "Bill to: Acme Logistics", "Item: freight, 3 pallets",
                    "Terms: net 30"], title="INVOICE", box=True).save(f"{OUT}/invoice_640x480.png")
    page(500, 700, ["Dear Ms. Patel,", "Thank you for your letter of", "the 3rd. We are pleased to", "confirm your appointment",
                    "as of next Monday.", "", "Yours sincerely,", "J. Ortega"]).save(f"{OUT}/letter_500x700.png")
    page(1400, 900, ["CAFE LUMEN", "1 x flat white      4.20", "1 x croissant       3.80", "card **** 4417",
                     "PAID  8.00", "thank you!"], title="RECEIPT").save(f"{OUT}/receipt_1400x900.png")
    print(sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
