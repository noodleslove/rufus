import os
import sys
from dotenv import load_dotenv, find_dotenv
import panel as pn

pn.extension()


class Utils:
    def __init__(self):
        pass

    def get_openai_api_key(self):
        _ = load_dotenv(find_dotenv())
        return os.getenv("OPENAI_API_KEY")

    def get_unstructured_api_key(self):
        _ = load_dotenv(find_dotenv())
        return os.getenv("UNSTRUCTURED_API_KEY")

    def get_unstructured_url(self):
        _ = load_dotenv(find_dotenv())
        return os.getenv("UNSTRUCTURED_API_URL")
