"""
The TextProcessor class provides methods to handle information from given text.

Methods:
find_date(text: str) -> str:
    Given a string of text, this method finds and returns the date in the text. If the date is not found,
    this method returns None.

find_money_or_cardinal(text: str) -> str:
    Given a string of text, this method finds and returns the amount mentioned in the text. If the amount is not found,
    this method returns None.

handle_text(text: str, label_id: int) -> str:
    Given a string of text and a label ID, this method returns the corresponding text for the given label ID:
    label_id 0: date
    label_id 1: amount
    label_id 2: other text
    If the label ID is not valid, this method returns None.
"""

import datefinder
import regex as re
import spacy
from app.constants import Regex, Labels


class TextProcessor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    async def find_date(self, text):
        cleaned_text = re.sub(Regex.CLEANED_TEXT, ' ', text, flags=re.IGNORECASE)
        matches = list(datefinder.find_dates(cleaned_text))
        if matches:
            for match in matches:
                date_str = match.strftime(Regex.DATE_FORMAT_1)
                captures = re.findall(Regex.CAPTURES_PATTERN, text)
                if int(re.findall(Regex.CAPTURES_PATTERN, date_str)[2]) == int(captures[0]):
                    return date_str
                else:
                    date_str = match.strftime(Regex.DATE_FORMAT_2)
                    return date_str
        else:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == Labels.DATE_LABEL:
                    return ent.text
                else:
                    str_pattern = Regex.DATE_PATTERN
                    for pattern in str_pattern:
                        matches = re.finditer(pattern, text)
                        if not matches:
                            return matches[0].group()


    async def find_money_or_cardinal(self, text):
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in Labels.AMOUNT_LABEL:
                return ent.text

    async def handle_text(self, text, label_id):
        strip_txt = text.strip()
        if len(strip_txt) != 0:
            if label_id == 2:
                return strip_txt
            elif label_id == 0:
                date = await self.find_date(strip_txt)
                return date
            elif label_id == 1:
                amount = await self.find_money_or_cardinal(strip_txt)
                return amount
