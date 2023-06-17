import pickle
import argparse
from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain

from constant import VECTOR_OUTPUT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="strikingly.com Q&A")
    parser.add_argument("question", type=str, help="Your question for strikingly.com")
    args = parser.parse_args()

    with open(VECTOR_OUTPUT_PATH, "rb") as f:
        store = pickle.load(f)

    chain = VectorDBQAWithSourcesChain.from_llm(
        llm=OpenAI(temperature=0), vectorstore=store
    )
    result = chain({"question": args.question})

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
