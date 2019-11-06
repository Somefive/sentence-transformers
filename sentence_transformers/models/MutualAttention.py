import torch
from torch import nn, Tensor
from typing import Union, Tuple, List, Iterable, Dict
from ..SentenceTransformer import SentenceTransformer
import logging

class MutualAttention(nn.Module):
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 num_labels: int,
                 concatenation_sent_rep: bool = True,
                 concatenation_sent_difference: bool = True,
                 concatenation_sent_multiplication: bool = False,
                 classification: bool = True):
        super(MutualAttention, self).__init__()
        self.model = model
        self.sentence_embedding_dimension = sentence_embedding_dimension
        self.num_labels = num_labels
        self.concatenation_sent_rep = concatenation_sent_rep
        self.concatenation_sent_difference = concatenation_sent_difference
        self.concatenation_sent_multiplication = concatenation_sent_multiplication
        num_vectors_concatenated = 0
        if concatenation_sent_rep:
            num_vectors_concatenated += 2
        if concatenation_sent_difference:
            num_vectors_concatenated += 1
        if concatenation_sent_multiplication:
            num_vectors_concatenated += 1
        logging.info("Mutual attention loss: #Vectors concatenated: {}".format(num_vectors_concatenated))
        self.classifier = nn.Linear(num_vectors_concatenated * sentence_embedding_dimension, num_labels)
        self.classification = classification

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        sentence_features = [self.model(sentence_feature) for sentence_feature in sentence_features]
        reps = []
        for features in sentence_features:
            token_embeddings = features['token_embeddings']
            input_mask = features['input_mask']
            input_mask_expanded = input_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            token_embeddings[input_mask_expanded == 0] = 0
            reps.append(token_embeddings)
        
        rep_a, rep_b = reps
        attn = torch.matmul(rep_a, rep_b.transpose(1,2))
        attn_a = nn.functional.softmax(attn.sum(dim=2, keepdim=True), dim=1).expand(rep_a.size())
        attn_b = nn.functional.softmax(attn.sum(dim=1, keepdim=True), dim=2).expand(rep_b.size())
        rep_a = (attn_a * rep_a).max(dim=1, keepdim=True)
        rep_b = (attn_b * rep_b).max(dim=1, keepdim=True)

        if self.classification:
            vectors_concat = []
            if self.concatenation_sent_rep:
                vectors_concat.append(rep_a)
                vectors_concat.append(rep_b)

            if self.concatenation_sent_difference:
                vectors_concat.append(torch.abs(rep_a - rep_b))

            if self.concatenation_sent_multiplication:
                vectors_concat.append(rep_a * rep_b)

            features = torch.cat(vectors_concat, 1)

            output = self.classifier(features)
            loss_fct = nn.CrossEntropyLoss()

            if labels is not None:
                loss = loss_fct(output, labels.view(-1))
                return loss
            else:
                return reps, output
        else:
            output = torch.cosine_similarity(rep_a, rep_b)
            loss_fct = nn.MSELoss()

            if labels is not None:
                loss = loss_fct(output, labels.view(-1))
                return loss
            else:
                return reps, output