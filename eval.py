"""
eval.py — InsightRAG Evaluation Script (RAGAS)

Ye script main.py ke RAG pipeline ko automatically test karta hai:
- 20 test questions + unke sahi (ground truth) answers
- System se jawab leta hai (main.py se import karke)
- RAGAS ke 4 metrics se score karta hai: Faithfulness, Answer Relevancy,
  Context Precision, Context Recall

IMPORTANT DESIGN CHOICE — Checkpointing:
Har question ko ALAG SE evaluate karke turant CSV mein save kiya jata hai
(poora batch ek saath evaluate karne ke bajaye). Isse:
- Agar beech mein crash ho jaye (laptop hang, bijli jaye), pehle wale
  questions ka result safe rehta hai
- Agla baar chalane par already-done questions skip ho jate hain (resume ho jata hai)
- Ek question ka evaluation fail ho to baaki questions pe asar nahi padta

NOTE: Judge LLM ab Groq (llama-3.3-70b, free, bahut fast) hai — pehle Ollama
(llama3.2, local, slow) use kiya tha jisme JSON parsing kabhi-kabhi fail hoti
thi. Groq bada model hai, isliye structured output zyada reliable hai aur
evaluation bahut fast hoga. Embeddings abhi bhi Ollama (nomic-embed-text) se
hi hain, kyunki wo free hai aur is task ke liye kaafi hai.
"""

import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from ragas import evaluate
from ragas.metrics import faithfulness, AnswerRelevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings

from main import setup_pipeline, answer_question

load_dotenv()  # .env file se GROQ_API_KEY load karega

# ================================
# Kitne questions is run mein test karne hain (already-done questions
# checkpoint file se automatically skip ho jayenge)
# ================================
NUM_QUESTIONS_TO_TEST = 10

CHECKPOINT_FILE = "eval_results.csv"

METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

# ================================
# Test dataset — PDF (Attention Is All You Need) padh ke banaye gaye 20 Q&A pairs
# ================================
TEST_QUESTIONS = [
    {
        "question": "What is the Transformer model based on?",
        "ground_truth": "The Transformer is a model architecture based entirely on attention mechanisms, dispensing with recurrence and convolutions entirely."
    },
    {
        "question": "What BLEU score does the Transformer achieve on the WMT 2014 English-to-German translation task?",
        "ground_truth": "The Transformer (big) model achieves a BLEU score of 28.4 on the WMT 2014 English-to-German translation task."
    },
    {
        "question": "What BLEU score does the Transformer achieve on the WMT 2014 English-to-French translation task?",
        "ground_truth": "The Transformer (big) model achieves a new single-model state-of-the-art BLEU score of 41.0 on the WMT 2014 English-to-French translation task."
    },
    {
        "question": "How long did it take to train the big Transformer model?",
        "ground_truth": "The big models were trained for 300,000 steps, which took 3.5 days on eight P100 GPUs."
    },
    {
        "question": "What is self-attention?",
        "ground_truth": "Self-attention, also called intra-attention, is an attention mechanism that relates different positions of a single sequence to compute a representation of that sequence."
    },
    {
        "question": "How many identical layers does the encoder consist of?",
        "ground_truth": "The encoder is composed of a stack of N = 6 identical layers."
    },
    {
        "question": "How many identical layers does the decoder consist of?",
        "ground_truth": "The decoder is also composed of a stack of N = 6 identical layers."
    },
    {
        "question": "What are the two sub-layers in each encoder layer?",
        "ground_truth": "Each encoder layer has two sub-layers: a multi-head self-attention mechanism and a simple, position-wise fully connected feed-forward network."
    },
    {
        "question": "What is the dimensionality of the model (d_model) used in the base Transformer?",
        "ground_truth": "The base model uses d_model = 512."
    },
    {
        "question": "What formula is used for Scaled Dot-Product Attention?",
        "ground_truth": "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V."
    },
    {
        "question": "Why is the dot product scaled by 1 over square root of dk?",
        "ground_truth": "For large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions with extremely small gradients, so scaling by 1/sqrt(dk) counteracts this effect."
    },
    {
        "question": "How many parallel attention heads does the Transformer use?",
        "ground_truth": "The Transformer uses h = 8 parallel attention heads."
    },
    {
        "question": "What are the dimensions dk and dv used for each attention head in the base model?",
        "ground_truth": "dk = dv = d_model / h = 64."
    },
    {
        "question": "What activation function is used in the position-wise feed-forward network?",
        "ground_truth": "The feed-forward network uses a ReLU activation between two linear transformations: FFN(x) = max(0, xW1+b1)W2+b2."
    },
    {
        "question": "What functions are used for positional encoding?",
        "ground_truth": "Sine and cosine functions of different frequencies are used for positional encoding."
    },
    {
        "question": "What optimizer was used to train the model?",
        "ground_truth": "The Adam optimizer was used, with beta1=0.9, beta2=0.98, and epsilon=10^-9."
    },
    {
        "question": "What regularization techniques were used during training?",
        "ground_truth": "Residual dropout and label smoothing (with a value of 0.1) were used during training."
    },
    {
        "question": "What dataset was used for the English-German translation training?",
        "ground_truth": "The standard WMT 2014 English-German dataset, consisting of about 4.5 million sentence pairs, was used."
    },
    {
        "question": "What hardware was used to train the models?",
        "ground_truth": "The models were trained on one machine with 8 NVIDIA P100 GPUs."
    },
    {
        "question": "What are the three ways attention is used in the Transformer model?",
        "ground_truth": "The Transformer uses encoder-decoder attention, encoder self-attention, and decoder self-attention (masked to preserve the auto-regressive property)."
    },
]


def load_already_done_questions():
    """Checkpoint file se pehle se evaluate ho chuke questions ki list nikalta hai."""
    if os.path.exists(CHECKPOINT_FILE):
        existing_df = pd.read_csv(CHECKPOINT_FILE)
        if "question" in existing_df.columns:
            return set(existing_df["question"].tolist())
    return set()


def append_result_to_csv(row_dict):
    """Ek question ka result turant CSV mein append karta hai (checkpointing)."""
    df_row = pd.DataFrame([row_dict])
    file_exists = os.path.exists(CHECKPOINT_FILE)
    df_row.to_csv(CHECKPOINT_FILE, mode="a", header=not file_exists, index=False)


def evaluate_single_question(judge_llm, judge_embeddings, question, answer, contexts, ground_truth):
    """
    Ek question ko RAGAS se evaluate karta hai (4 metrics). Isolated hai —
    agar ye fail ho, sirf isi question ka result missing rahega, baaki chalte rahenge.
    """
    eval_dataset = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
        "ground_truth": [ground_truth],
    })

    answer_relevancy_metric = AnswerRelevancy(strictness=1)
    metrics = [faithfulness, answer_relevancy_metric, context_precision, context_recall]

    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=judge_llm,
            embeddings=judge_embeddings,
            run_config=RunConfig(timeout=600, max_workers=2, max_retries=2),
        )
        df = result.to_pandas()
        scores = {m: df.iloc[0].get(m, None) for m in METRIC_NAMES}
    except Exception as e:
        print(f"    !! Evaluation fail hua is question ke liye: {e}")
        scores = {m: None for m in METRIC_NAMES}

    return scores


def main():
    print("=" * 60)
    print("InsightRAG Evaluation — RAGAS")
    print("=" * 60)

    print("\nQdrant se connect ho raha hu (agar data pehle se store hai, dobara nahi banega)...")
    client = setup_pipeline()

    already_done = load_already_done_questions()
    if already_done:
        print(f"\nCheckpoint mila: {len(already_done)} questions pehle se evaluate ho chuke hain, skip karenge.")

    questions_subset = TEST_QUESTIONS[:NUM_QUESTIONS_TO_TEST]
    remaining = [q for q in questions_subset if q["question"] not in already_done]

    if not remaining:
        print("\nSaare selected questions pehle se evaluate ho chuke hain. NUM_QUESTIONS_TO_TEST badhao naye test karne ke liye.")
        return

    print(f"\n{len(remaining)} naye questions evaluate honge (out of {len(questions_subset)} total selected).\n")

    print("Judge LLM (Groq — fast, free) setup ho raha hai...")
    if not os.getenv("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY nahi mili. Check karo .env file mein sahi se daali hai ya nahi.")
        return

    judge_llm = LangchainLLMWrapper(
        ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=os.getenv("GROQ_API_KEY"))
    )
    judge_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))
    print("Setup ho gaya. Evaluation shuru...\n")

    for i, item in enumerate(remaining, start=1):
        print(f"[{i}/{len(remaining)}] Poochh raha hu: {item['question']}")
        result = answer_question(client, item["question"])

        print(f"    -> Confidence: {result['confidence']:.2f} | Answered: {result['answered']}")
        print("    -> Ab RAGAS se evaluate ho raha hai...")

        scores = evaluate_single_question(
            judge_llm, judge_embeddings,
            item["question"], result["answer"], result["contexts"], item["ground_truth"]
        )

        row = {
            "question": item["question"],
            "answer": result["answer"],
            "ground_truth": item["ground_truth"],
            **scores,
        }
        append_result_to_csv(row)

        score_summary = ", ".join(
            f"{m}={scores[m]:.2f}" if scores[m] is not None else f"{m}=N/A"
            for m in METRIC_NAMES
        )
        print(f"    -> Saved. Scores: {score_summary}\n")

    # ================================
    # Final summary — poora checkpoint file padh ke overall report banate hain
    # ================================
    print("=" * 60)
    print("OVERALL REPORT (saare saved results milake)")
    print("=" * 60)

    full_df = pd.read_csv(CHECKPOINT_FILE)
    total_rows = len(full_df)

    for metric_name in METRIC_NAMES:
        if metric_name in full_df.columns:
            valid_count = full_df[metric_name].notna().sum()
            avg_score = full_df[metric_name].mean(skipna=True)
            if pd.notna(avg_score):
                print(f"Average {metric_name}: {avg_score:.2f}  (based on {valid_count}/{total_rows} valid scores)")
            else:
                print(f"Average {metric_name}: N/A  (0/{total_rows} valid scores)")

    print(f"\nPoora result yahan hai: {CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()