# Speaker Role Extractor using GPT-4

import fitz  # PyMuPDF
import openai
import ast
import re

openai.api_key = "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"

# Step 1: Extract First Few Pages of Transcript
def extract_intro_text(pdf_path, max_pages=2):
    doc = fitz.open(pdf_path)
    intro = ""
    for i in range(min(max_pages, len(doc))):
        intro += doc[i].get_text()
    return intro

# Step 2: Ask GPT-4 to Extract Speaker Roles
def extract_roles_with_gpt(intro_text):
    system_msg = "You are a helpful assistant parsing a financial earnings call transcript."

    user_msg = f"""
    The following is the beginning of an earnings call transcript. Extract all speaker names and their roles.
    Return the result as a Python dictionary like:
    {{ "Full Name": "Role" }}

    Text:
    {intro_text}
    """

    response = openai.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=0
    )

    result_text = response.choices[0].message.content
    speaker_dict = safe_parse_dict(result_text)
    return speaker_dict

# Safe dictionary parsing

def safe_parse_dict(text):
    # Remove markdown/code fences
    text = re.sub(r"```(?:python)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return ast.literal_eval(match.group())
    except Exception as e:
        print("❌ Failed to parse GPT response:", e)
        return {}

# Example usage
if __name__ == "__main__":
    transcript_path = "bel_transcript.pdf"
    intro = extract_intro_text(transcript_path)
    speaker_roles = extract_roles_with_gpt(intro)
    print("Dict:",speaker_roles)
    print("\n✅ Speaker Roles Extracted:")
    for name, role in speaker_roles.items():
        print(f"- {name}: {role}")
