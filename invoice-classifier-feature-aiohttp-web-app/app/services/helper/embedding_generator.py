"""
This module contains the EmbeddingsGenerator class responsible for generating BERT embeddings from given text using a pre-trained transformer.
"""

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from app.constants import Path


class EmbeddingsGenerator:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(Path.DISTILBERT_PATH)
        self.model = AutoModel.from_pretrained(Path.DISTILBERT_PATH)

    def __mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def generate_text_embeddings(self, inputs, batch_size=32):
        """ The method generates BERT embeddings from given text using pre-trained transformer """

        embeddings = []
        for i in tqdm(range(0, len(inputs), batch_size)):
            batch_text = inputs[i:i + batch_size]
            encoded_input = self.tokenizer(batch_text, padding=True, truncation=True, return_tensors='pt', max_length=400)
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            sentence_embeddings_chunk = self.__mean_pooling(model_output, encoded_input['attention_mask'])
            embeddings_tensor = sentence_embeddings_chunk.cpu().numpy().tolist()
            embeddings.extend(embeddings_tensor)

        return embeddings

