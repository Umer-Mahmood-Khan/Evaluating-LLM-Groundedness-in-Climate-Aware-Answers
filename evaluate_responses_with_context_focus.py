import json
import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Load stored responses
with open("responses.json", "r", encoding="utf-8") as f:
    responses_data = json.load(f)

# Initialize GPT model
llm = ChatOpenAI(model_name="gpt-4.1-nano", temperature=0)

# Evaluation prompt (emphasizing Use of Retrieved Content)
eval_template = """
You are an evaluator.

Evaluate the answer using these metrics:
- Accuracy (1-5)
- Completeness (1-5)
- Clarity (1-5)
- Relevance (1-5)
- **Use of Retrieved Content (1-5)**

For 'Use of Retrieved Content':
- Give 1 if the answer does not refer to or use information from the retrieved context.
- Give 5 if the answer is clearly based on, references, or depends on retrieved context.

Provide scores as:
Accuracy: X
Completeness: X
Clarity: X
Relevance: X
Use of Retrieved Content: X

---

Question:
{question}

Retrieved Chunks:
{chunks}

Answer to Evaluate:
{answer}
"""

eval_prompt = PromptTemplate(
    input_variables=["question", "chunks", "answer"],
    template=eval_template
)

evaluation_chain = LLMChain(
    llm=llm,
    prompt=eval_prompt
)

results = []

# Loop over each record and evaluate both answers
for record in responses_data:
    question = record["question"]
    chunks_combined = "\n---\n".join(record["retrieved_chunks"])

    for response_type in ["pure_chatgpt_response", "rag_response"]:
        answer = record[response_type]

        # Evaluate answer
        eval_output = evaluation_chain.run({
            "question": question,
            "chunks": chunks_combined,
            "answer": answer
        })

        # Save results
        results.append({
            "question": question,
            "response_type": response_type,
            "answer": answer,
            "evaluation_scores": eval_output
        })

# Save evaluation results
with open("evaluation_results_context_focus.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("Evaluation complete. Results saved to evaluation_results_context_focus.json.")
