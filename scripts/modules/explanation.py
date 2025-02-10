import pandas as pd

from torch import device
from torch.nn.functional import softmax
from transformers import AutoConfig, BertTokenizerFast

from typing import Callable, List, Union

import sys
sys.path.append("vendor/Transformer-Explainability")

from BERT_explainability.modules.BERT.BertForSequenceClassification import BertForSequenceClassification
from BERT_explainability.modules.BERT.ExplanationGenerator import Generator


class BertExplainer:

    # constructor
    def __init__(
        self, 
        classifier_path: str, 
        device: Union[int, str, device]
    ):
        self.__classifier_path = classifier_path
        self.__device = device

        self.__model = BertForSequenceClassification.from_pretrained(classifier_path).to(device)
        self.__tokenizer = BertTokenizerFast.from_pretrained(classifier_path) # use BertTokenizerFast to be able to access char offsets
        self.__explanation_generator = Generator(self.__model) # use Layer-wise Relevance Propagation to generate explanations

        self.__config = AutoConfig.from_pretrained(classifier_path)
        self.__labels = list(self.__config.to_dict()["label2id"])
    
    # getters
    def get_labels(self) -> List[str]:
        return self.__labels

    # main methods
    def explain(self, text: str) -> dict:
        # tokenize text to create input
        encoded_text = self.__tokenizer(text, return_tensors='pt', return_offsets_mapping=True)
        
        # keep additional tokenization info
        input_ids = encoded_text['input_ids'].to(self.__device)
        tokens = self.__tokenizer.convert_ids_to_tokens(input_ids.flatten())
        
        offset_mapping = [(int(offsets[0]), int(offsets[1])) for offsets in encoded_text['offset_mapping'][0]]

        attention_mask = encoded_text['attention_mask'].to(self.__device)

        # get classification output
        output = softmax(self.__model(input_ids=input_ids, attention_mask=attention_mask)[0], dim=-1)
        label_id = output.argmax(dim=-1).item()
        label = self.__labels[label_id]

        # add (normalized) explanations
        explanations = self.__explanation_generator.generate_LRP(input_ids=input_ids, attention_mask=attention_mask, start_layer=0)[0]
        explanations = (explanations - explanations.min()) / (explanations.max() - explanations.min())
        
        return {
            "text": text,
            "tokens": tokens,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "offset_mapping": offset_mapping,

            "classification": label,
            "probabilities": [float(x) for x in output[0]],

            "explanations": explanations.tolist()
        }
    
    def __to_redscale(label: str, score: float) -> str:
        lig = 100 - int(50 * score)
        return f"hsl(341, 100%, {lig}%)"
    
    def to_html(
        self, 
        output: dict, 
        to_color: Callable[[str, float], str] = __to_redscale
    ) -> str:
        html = ""

        # extract text and label
        text = output["text"]
        label = output["classification"]

        # added colored spans per token
        current_offset = 0
        for i, offsets in enumerate(output["offset_mapping"]):
            # get offsets
            start = offsets[0]
            stop = offsets[1]

            if start < stop:
                # add untokenized chars
                for j in range(current_offset, start):
                    html += text[j]

                # get and style tokens
                token = text[start:stop]
                explanation_score = output["explanations"][i]
                html += f'<span style="background-color: {to_color(label, explanation_score)}">{token}</span>'

                # set 
                current_offset = stop
        return html
    
    def explain_as_html_table(
            self, 
            df: pd.DataFrame, 
            to_color: Callable[[str, float], str] = __to_redscale
    ) -> str:
        html = "<table>\n"
        html += "\t<thead>\n"
        html += "\t\t<tr>\n"
        html += "\t\t\t<th>source</th><th>n</th><th>text</th><th>label</th>"
        for label in self.__labels:
            html += f"<th>{label}</th>\n"
        html += "\t\t</tr>\n"
        html += "\t</thead>\n"
        html += "\t<tbody>\n"
        #for text in texts:
        for i, row in df.iterrows():
            text = row["text"]
            output = self.explain(text)

            html += "\t\t<tr>\n"
            html += "\t\t\t<td>"
            html += str(row["source"])
            html += "</td><td>"
            html += str(row["n"])
            html += "</td><td>"
            html += self.to_html(output, to_color)
            html += "</td><td>"
            html += output["classification"]
            html += "</td>"
            for x in output["probabilities"]:
                html += "<td>"
                html += f"{x:.2%}"
                html += "</td>"
            html += "\t\t</tr>\n"
        html += "\t</tbody>\n"
        html += "</table>"
        return html