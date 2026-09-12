from dotenv import find_dotenv, load_dotenv

from .directories import secrets


def find_reusable_llm_provider_env():
    return find_dotenv(secrets(".env"))


def load_reusable_llm_provider_env():
    _ = load_dotenv(find_reusable_llm_provider_env())
