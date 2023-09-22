from transformers import AutoTokenizer, AutoModelForTokenClassification
import os
import torch
from src.load_data import load_data_NER

class NER():
    def __init__(self):
        model_path = os.path.abspath("model")
        self.MAXLEN = 128
        self.tokenize = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(model_path)
        
    def tagging_NER(self, text):
        if torch.cuda.is_available():
            device = "cuda"
            print("Run on GPU")
        else:
            device = "cpu"
            print("Run on CPU")
        
        input = self.tokenize(text, padding="max_length", truncation=True, max_length=self.MAXLEN, return_tensors="pt")
        id2label = load_data_NER()

        # move to gpu
        ids = input["input_ids"].to(device)
        mask = input["attention_mask"].to(device)
        # forward pass
        self.model.to(device)
        outputs = self.model(ids, mask)
        logits = outputs[0]

        active_logits = logits.view(-1, len(id2label)) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size*seq_len,) - predictions at the token level

        

        tokens = self.tokenize.convert_ids_to_tokens(ids.squeeze().tolist())
        token_predictions = [id2label[i] for i in flattened_predictions.cpu().numpy()]
        wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

        word_level_predictions = []
        for pair in wp_preds:
            if (pair[0].startswith(" ##")) or (pair[0] in ['[CLS]', '[SEP]', '[PAD]']):
                # skip prediction
                continue
            else:
                word_level_predictions.append(pair[1])

        # we join tokens, if they are not special ones
        str_rep = " ".join([t[0] for t in wp_preds if t[0] not in ['[CLS]', '[SEP]', '[PAD]']]).replace(" ##", "")

        return str_rep, word_level_predictions