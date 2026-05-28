from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
from src.config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def get_model(repo_id="DeepSeek", **kwargs):
    if repo_id in ("ChatGPT", "DeepSeek"):
        if repo_id == "DeepSeek":
            kwargs.setdefault("base_url", DEEPSEEK_BASE_URL)
            kwargs.setdefault("api_key", DEEPSEEK_API_KEY)
            kwargs.setdefault("model", DEEPSEEK_MODEL)
        chat_model = ChatOpenAI(temperature=0, **kwargs)
    else:
        raise ValueError(
            f"Unsupported model: {repo_id}. "
            f"Supported: 'DeepSeek', 'ChatGPT'. "
            f"HuggingFace Hub models are not supported in this version."
        )
    return chat_model


def basic_chain(model=None, prompt=None):
    if not model:
        model = get_model()
    if not prompt:
        prompt = ChatPromptTemplate.from_template("Tell me the most noteworthy books by the author {author}")

    chain = prompt | model
    return chain


def main():
    load_dotenv()

    prompt = ChatPromptTemplate.from_template("Tell me the most noteworthy books by the author {author}")
    chain = basic_chain(prompt=prompt) | StrOutputParser()

    results = chain.invoke({"author": "William Faulkner"})
    print(results)


if __name__ == '__main__':
    main()
