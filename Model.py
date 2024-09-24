from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class Model():
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.QandAGenerator = T5ForConditionalGeneration.from_pretrained("./models/questionandanswer").to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("./models/tokeniser")
        self.DistracterGen = T5ForConditionalGeneration.from_pretrained("./models/distrackerGen").to(self.device)

    def genQandA(self, context):
        context_token = self.tokenizer.encode_plus(
            f"[MASK] <sep> {context}",
            max_length=512,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt' 
        )

        input_ids = context_token['input_ids'].to(self.device)
        attention_mask = context_token['attention_mask'].to(self.device)

        output = self.QandAGenerator.generate(input_ids=input_ids, attention_mask=attention_mask)

      
        output = self.tokenizer.decode(output[0], skip_special_tokens=True).split("<sep>")
        return {"answer": output[0], "question": output[1]}

    def distracter(self, answer, question, context):
        
        context_token = self.tokenizer.encode_plus(
            f"{answer} <sep> {question} {context}",
            max_length=512,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt' 
        )

       
        input_ids = context_token['input_ids'].to(self.device)
        attention_mask = context_token['attention_mask'].to(self.device)

       
        output = self.DistracterGen.generate(input_ids=input_ids, attention_mask=attention_mask)

       
        output = self.tokenizer.decode(output[0], skip_special_tokens=True).split("<sep>")
        return output

    def gen(self, context):
        
        data = self.genQandA(context)
        data['distracter'] = self.distracter(data['answer'], data['question'], context)
        return data
