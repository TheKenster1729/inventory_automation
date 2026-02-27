import os
import json
import base64
from PIL import Image
from google import genai

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

PROMPT = """You are an inventory data extraction assistant. I am sending you a photo of a box of
laboratory/medical supplies. Extract the following fields. Return ONLY a valid JSON object
(no markdown formatting) with exactly these keys:

{
  "item_name": "full product name (e.g. 'Kimtech A7 Cleanroom Lab Coat')",
  "manufacturer": "company that made the product (e.g. 'Ansell', 'Kimberly-Clark')",
  "lot_number": "lot or batch number printed on the label",
  "quantity": "number of units in the box (e.g. '30 PCS')"
}

If a field is not visible or legible, set its value to null.
"""


def analyze_image(image_path: str) -> dict:
    image = Image.open(image_path)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=[PROMPT, image],
    )

    text = response.text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]

    return json.loads(text)

def lambda_handler(event, context):
    body = json.loads(event["body"])
    image_b64 = body["image"]

    tmp_path = "/tmp/capture.jpg"
    with open(tmp_path, "wb") as f:
        f.write(base64.b64decode(image_b64))

    result = analyze_image(tmp_path)

    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(result),
    }
