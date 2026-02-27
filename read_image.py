import os
import json
import glob
from google import genai
from PIL import Image

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

BATCH_PROMPT = """You are an inventory data extraction assistant. I am sending you {count} photos of 
boxes of laboratory/medical supplies. For EACH image, extract the following fields. Return ONLY a 
valid JSON array (no markdown formatting) with one object per image, in the same order as the images.
Each object must use exactly these keys:

{{
  "item_name": "full product name (e.g. 'Kimtech A7 Cleanroom Lab Coat')",
  "manufacturer": "company that made the product (e.g. 'Ansell', 'Kimberly-Clark')",
  "lot_number": "lot or batch number printed on the label",
  "quantity": "number of units in the box (e.g. '30 PCS')"
}}

If a field is not visible or legible, set its value to null.
"""


def process_folder(folder_path: str) -> list[dict]:
    """Send all images in a folder in a single Gemini request and return inventory records."""
    image_files = sorted(
        glob.glob(os.path.join(folder_path, "*.jpg"))
        + glob.glob(os.path.join(folder_path, "*.jpeg"))
        + glob.glob(os.path.join(folder_path, "*.png"))
    )

    if not image_files:
        print(f"No images found in '{folder_path}'.")
        return []

    print(f"Found {len(image_files)} images. Sending as a single batch request...\n")

    # Build the content list: prompt + all images interleaved with filenames
    contents = [BATCH_PROMPT.format(count=len(image_files))]
    filenames = []
    for path in image_files:
        filename = os.path.basename(path)
        filenames.append(filename)
        contents.append(f"Image {len(filenames)}: {filename}")
        contents.append(Image.open(path))

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=contents,
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    records = json.loads(text)

    # Attach source filenames
    for i, rec in enumerate(records):
        rec["source_file"] = filenames[i] if i < len(filenames) else "unknown"

    return records


if __name__ == "__main__":
    INPUT_FOLDER = os.path.join(os.path.dirname(__file__), "Test Images")
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "inventory_results.json")

    records = process_folder(INPUT_FOLDER)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\nResults written to {OUTPUT_FILE}")
    print(json.dumps(records, indent=2))
