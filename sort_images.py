import os
import shutil
import base64
import ast
import anthropic

# --------------------------------------------------------------------
# 1. Setup: Create your 81 subfolders in 'outputs'
# --------------------------------------------------------------------
OUTPUT_ROOT = "outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# Create all possible 4-digit ternary codes from 0000 to 2222
for s in range(3):      # shading
    for n in range(3):  # number
        for c in range(3):  # color
            for h in range(3):  # shape
                folder_name = f"{s}{n}{c}{h}"
                os.makedirs(os.path.join(OUTPUT_ROOT, folder_name), exist_ok=True)

# --------------------------------------------------------------------
# 2. Claude API details (fill in your actual API key and model)
# --------------------------------------------------------------------
api_key = "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"

SYSTEM_PROMPT = (
    "You are given an image containing exactly one card from the SET game. "
    "Return a 4-element array of integers corresponding to the card as follows: "
    "1) Shading (0=solid, 1=striped, 2=outlined). "
    "4) Number (0=one, 1=two, 2=three). "
    "2) Color (0=red, 1=purple, 2=green). "
    "3) Shape (0=oval, 1=squiggle, 2=diamond). "
    "Reply only with the final array of 4 points. If the image is not a card, respond only with '0'."
)

client = anthropic.Anthropic(api_key=api_key)

def call_claude(client, image_bytes, media_type):
    """
    Call the Claude API with a single image (in bytes).
    Return the response text.
    """
    image_data = base64.b64encode(image_bytes).decode("utf-8")

    message = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=200,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                ],
            }
        ]
    )
    return message.content[0].text.strip()

# --------------------------------------------------------------------
# 3. Helper function to parse the response from Claude
# --------------------------------------------------------------------
def parse_claude_response(response_str):
    """
    If response is '0', return None.
    Otherwise, parse the 4-element array, e.g. '[0, 0, 1, 2]',
    and return the corresponding 4-digit string, e.g. '0012'.
    """
    if response_str == "0":
        return None
    try:
        arr = ast.literal_eval(response_str)  # Should be something like [0, 0, 1, 2]
        if len(arr) == 4 and all(x in [0,1,2] for x in arr):
            code = "".join(str(x) for x in arr)
            return code
    except Exception:
        pass
    # If parsing fails or something unexpected, treat as invalid
    return None

# --------------------------------------------------------------------
# 4. Main logic: read images from 'inputs', call Claude, sort them
# --------------------------------------------------------------------
def process_images(input_folder="inputs"):
    # Ensure input folder exists
    if not os.path.isdir(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return

    # List all files in the input folder (you can refine if needed)
    all_files = os.listdir(input_folder)

    for filename in all_files:
        # Skip non-image files as needed
        # Here we just do a simple extension check for demonstration
        lower_name = filename.lower()
        if not (lower_name.endswith(".jpg") or lower_name.endswith(".jpeg")
                or lower_name.endswith(".png") or lower_name.endswith(".gif")
                or lower_name.endswith(".bmp")):
            continue

        # Full path
        file_path = os.path.join(input_folder, filename)

        # Read bytes
        with open(file_path, "rb") as f:
            image_bytes = f.read()

        # Guess media type from extension
        if lower_name.endswith(".png"):
            media_type = "image/png"
        elif lower_name.endswith(".jpg") or lower_name.endswith(".jpeg"):
            media_type = "image/jpeg"
        elif lower_name.endswith(".gif"):
            media_type = "image/gif"
        else:
            media_type = "image/bmp"

        # Call Claude
        response = call_claude(client, image_bytes, media_type)
        # Parse response
        code = parse_claude_response(response)

        # If code is valid (e.g. "0012"), copy the image to outputs/0012/
        if code is not None:
            dest_folder = os.path.join(OUTPUT_ROOT, code)
            # Safety check: folder should exist from our earlier creation
            if os.path.isdir(dest_folder):
                dest_path = os.path.join(dest_folder, filename)
                shutil.copy2(file_path, dest_path)
                print(f"Copied {filename} -> {code}/")
            else:
                print(f"Unexpected: folder {dest_folder} does not exist.")
        else:
            # "0" or invalid means do nothing
            print(f"Skipping {filename}, response={response}")

if __name__ == "__main__":
    process_images("inputs")
