import json
import os

def save_responses_as_json(
    filename,
    question,
    retrieved_chunks,
    pure_response,
    rag_response
):
    # Prepare retrieved chunks as combined text
    chunks_text = [doc.page_content for doc in retrieved_chunks]

    # Create record
    record = {
        "question": question,
        "retrieved_chunks": chunks_text,
        "pure_chatgpt_response": pure_response,
        "rag_response": rag_response
    }

    # Load existing file or create new
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    # Append new record
    data.append(record)

    # Save
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
