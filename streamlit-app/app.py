import os
import base64
import ast
import streamlit as st
import anthropic
from algorithm import find_line_indices

# ---------------------------------------------
# 1) Session State Initialization
# ---------------------------------------------
if "points" not in st.session_state:
    st.session_state["points"] = None

if "results" not in st.session_state:
    st.session_state["results"] = None

if "highlighted_cards" not in st.session_state:
    st.session_state["highlighted_cards"] = set()

# ---------------------------------------------
# 2) Claude System Prompt
# ---------------------------------------------
SYSTEM_PROMPT = (
    "You are given an image containing exactly 12 cards from the SET game. "
    "For each card, return a 4-element array of integers: "
    "1) Shading (0=solid, 1=striped, 2=outlined). "
    "2) Color (0=red, 1=purple, 2=green). "
    "3) Shape (0=diamond, 1=ovals, 2=squiggle). "
    "4) Number (0=one, 1=two, 2=three). "
    "Reply only with the final array of 12 points and no extra text."
)

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
def point_to_image_path(point):
    """Convert a card point (list of 4 ints) to its image path."""
    shading, color, shape, number = point
    filename = f"{shape}{number}.png"
    return os.path.join("cards", str(shading), str(color), filename)

def display_cards(points):
    """Display the grid of 12 cards with a refined highlight style."""
    st.subheader("Identified Cards")
    n_rows, n_cols = 3, 4
    idx = 0
    for row in range(n_rows):
        cols = st.columns(n_cols)
        for col in range(n_cols):
            if idx < len(points):
                card_point = points[idx]
                image_path = point_to_image_path(card_point)
                try:
                    with open(image_path, "rb") as f:
                        img_data = f.read()
                    b64_str = base64.b64encode(img_data).decode("utf-8")
                except FileNotFoundError:
                    b64_str = ""
                # Use a subtle red glow with a slight scale-up for highlighted cards
                if idx in st.session_state["highlighted_cards"]:
                    card_style = (
                        "border: 4px solid red; "
                        "box-shadow: 0 0 10px rgba(255,0,0,0.8); "
                        "transform: scale(1.05); "
                        "transition: transform 0.2s ease-in-out; "
                        "padding: 4px;"
                    )
                else:
                    card_style = (
                        "border: 1px solid #ccc; "
                        "transition: transform 0.2s ease-in-out; "
                        "padding: 4px;"
                    )
                with cols[col]:
                    st.markdown(
                        f"""
                        <div style="{card_style}">
                            <img src="data:image/png;base64,{b64_str}" style="width:100%;" />
                            <div style="text-align:center; font-size:0.9em; color:#888;">{card_point}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                idx += 1

def highlight_set(triple):
    """Callback to update highlighted cards with the indices from a SET."""
    st.session_state["highlighted_cards"].clear()
    st.session_state["highlighted_cards"].update(triple)

def display_results(results):
    """Display each found SET with an expander and a neatly placed highlight button."""
    st.subheader("Results")
    attr_maps = [
        {0: "solid", 1: "striped", 2: "outlined"},
        {0: "red", 1: "purple", 2: "green"},
        {0: "diamond", 1: "oval", 2: "squiggle"},
        {0: "single", 1: "double", 2: "triple"},
    ]
    for i, (triple, attr_values) in enumerate(results, start=1):
        # Generate an explanation for the SET
        if all(val == -1 for val in attr_values):
            explanation = "everything different"
        else:
            explanation = ", ".join(
                attr_maps[attr_idx][val]
                for attr_idx, val in enumerate(attr_values) if val != -1
            )
        # Create two columns: one for the expander and one for the highlight button
        cols = st.columns([4, 1])
        with cols[0]:
            with st.expander(f"SET {i} = {triple}", expanded=False):
                st.write(explanation)
        with cols[1]:
            st.button(
                "Highlight",
                key=f"highlight_set_{i}",
                on_click=highlight_set,
                args=(triple,),
            )

def call_claude(client, image_bytes, media_type):
    """Call the Claude API and return the response message."""
    with st.spinner("Calling Claude..."):
        image_data = base64.b64encode(image_bytes).decode("utf-8")
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
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
    return message

# ---------------------------------------------
# Main App Function
# ---------------------------------------------
def main():
    st.title("SET Solver")

    # Ask for Claude API key
    api_key = st.text_input("Enter your Claude API key:", type="password", key="api_key_field")
    if not api_key:
        st.warning("Please enter your API key.")
        st.stop()
    client = anthropic.Anthropic(api_key=api_key)

    # File uploader for the SET game image
    uploaded_file = st.file_uploader("Upload a picture of the 12 cards", type=["png", "jpg", "jpeg"], key="file_uploader")
    image_bytes = None
    media_type = None
    if uploaded_file:
        image_bytes = uploaded_file.read()
        media_type = uploaded_file.type

    # Process button: call Claude once
    if image_bytes is not None and st.button("Process", key="process_btn"):
        message = call_claude(client, image_bytes, media_type)
        raw_text = message.content[0].text if len(message.content) > 0 else ""
        try:
            points_as_tuples = ast.literal_eval(raw_text)
            points = [list(tup) for tup in points_as_tuples]
            st.session_state["points"] = points
            st.session_state["results"] = find_line_indices(points)
        except Exception as e:
            st.error(f"Error parsing array from Claude: {e}")

    # Display the grid of cards
    if st.session_state["points"]:
        display_cards(st.session_state["points"])

    # Display possible SETs with highlight buttons
    if st.session_state["results"]:
        display_results(st.session_state["results"])

if __name__ == "__main__":
    main()
