from typing import List


class Token:
    def __init__(self, word_form: str, **kwargs):
        self.word_form = word_form

        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

    @staticmethod
    def get_root_word_form():
        return "#$%^&"

    @staticmethod
    def root_token():
        return Token(word_form=Token.get_root_word_form())

    def is_root_token(self):
        return self.word_form == self.get_root_word_form()


class Sentence:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.tokens_with_root = [Token.root_token()] + tokens
