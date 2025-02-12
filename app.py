import os
import base64
import ast
import streamlit as st
import anthropic
from algorithm import find_line_indices  # your custom logic

# client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = (
    "I will upload an image of 12 cards from the SET game. "
    "Your job is to interpret the image and return an array of 12 points representing the cards."
    "You will map each card to a point as follows:"
    "Each point will have 4 coordinates, each of which can take values 0,1,2. "
    "The first coordinate represents shape: 0 oval, 1 squiggle, 2 diamonds. "
    "The second coordinate represents color: 0 Red, 1 Purple, 2 Green. "
    "The third coordinate represents number: 0 one, 1 two, 2 three. "
    "The fourth coordinate represents shading: 0 solid, 1 striped, 2 outlined. "
    "Reply only with the final array of 12 points."
)

def main():
    st.title("SET Solver")

    api_key = st.text_input("Enter your Claude API key:", type="password")
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
        # ... proceed with the rest of your code
    else:
        st.warning("Please enter your API key above.")
        st.stop()

    # Choose how to get the image: "File Upload" or "Camera"
    choice = st.radio("Image source:", ["Upload a file", "Take a photo"])

    # We’ll store the final image bytes (if any) in this variable
    image_bytes = None

    # -----------------------------------------
    # A) Upload a file from the local computer
    # -----------------------------------------
    if choice == "Upload a file":
        uploaded_file = st.file_uploader("Upload a snapshot of the 12 cards", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image_bytes = uploaded_file.read()
            # Optionally show a preview
            st.image(image_bytes, caption="Uploaded image", use_container_width=True)
            media_type = uploaded_file.type

    # -----------------------------------------
    # B) Take a photo from the user’s camera
    # -----------------------------------------
    else:
        picture = st.camera_input("Take a snapshot of the 12 cards")
        if picture:
            # picture is a Streamlit UploadedFile object
            # calling getvalue() returns its binary content
            image_bytes = picture.getvalue()
            # st.image(image_bytes, caption="Snapshot from camera", use_container_width=True)
            media_type = picture.type
    # Only show the "Process" button if we have an image
    if image_bytes is not None:
        if st.button("Identify the SETs"):
            with st.spinner("Thinking..."):
                # Base64-encode the image
                image_data = base64.b64encode(image_bytes).decode("utf-8")

                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",  # or your Claude model
                    max_tokens=1000,
                    temperature=0,
                    system=SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Here is the image of 12 SET cards. Respond only with the final array."
                                    "After you think it through, go back and confirm your answer by matching the cards with the coordinates you provided."
                                },
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_data,
                                    },
                                },
                            ]
                        }
                    ]
                )

            # # Show raw response
            # st.subheader("Claude’s Raw Response")
            # st.write(message.content)

            # Attempt to parse the array
            if len(message.content) > 0:
                raw_text = message.content[0].text
                try:
                    points_as_tuples = ast.literal_eval(raw_text)
                    points = [list(tup) for tup in points_as_tuples]

                    # st.subheader("Parsed Points")
                    # st.write(points)

                    # Run find_line_indices
                    st.subheader("Results")
                    results = find_line_indices(points)
                    if results:
                        for i, triple in enumerate(results, start=1):
                            st.write(f"**SET {i}:** {triple}")
                    else:
                        st.write("No SETs found.")
                except Exception as e:
                    st.error(f"Could not parse array from response: {e}")

if __name__ == "__main__":
    main()
