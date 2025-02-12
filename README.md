# SET Solver

This project provides a web application (currently uses **Streamlit**) that identifies 12 cards from the **SET** card game and returns a **SET**. It uses **Anthropic’s Claude** LLM to interpret a user-supplied image (either uploaded or taken with a camera) and then computes possible matches via custom algorithm implemented in Python. The app is deployed at [https://set-solver.streamlit.app/](https://set-solver.streamlit.app/).

## Features

- **Upload or Capture**: Users can upload a photo of 12 SET cards **or** snap one from their camera.  
- **Claude Integration**: The app sends the image to Claude via Anthropic’s API, requesting a parsed list of card attributes.  
- **Line Finding Algorithm**: After retrieving card attributes, the code runs a custom line finding algorithm to process the data and display potential sets in a user-friendly format.

## Quickstart

1. **Clone the repo** and change to the project directory:

   ```bash
   git clone https://github.com/nakkstar123/set-solver.git
   cd set-solver
   ```

2. **Set up a virtual environment** (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

   By default, it will be served at [http://localhost:8501](http://localhost:8501).

## TODO

- [ ] Figure out bug with camera capture (sometimes the photo does not initialize or raises an error).  
- [ ] Better error handling for ambiguous cases (e.g., if Claude doesn't parse the image correctly).  
- [ ] Deal with rotated photos (add orientation correction if needed).  
- [ ] More readable outputs (improve how lines and sets are displayed).  
- [ ] Add Claude authentication so each user can enter their own key, rather than using a single developer key.
