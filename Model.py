from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.optim import Adam
from tqdm import tqdm

class Model():
    def __init__(self) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.QandAGenerator = T5ForConditionalGeneration.from_pretrained("./models/questionandanswer").to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained("./models/tokeniser")
        self.DistracterGen = T5ForConditionalGeneration.from_pretrained("./models/distrackerGen").to(self.device)

    def genQandA(self, context,num_QA):
        context_token = self.tokenizer.encode_plus(
            context,
            max_length=512,
            return_attention_mask=True,
            padding='max_length',
            truncation=True,
            return_tensors='pt' 
        )
        self.QandAGenerator.eval()

        input_ids = context_token['input_ids'].to(self.device)
        attention_mask = context_token['attention_mask'].to(self.device)

        output = self.QandAGenerator.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=128,
    num_return_sequences=num_QA,
    num_beams=num_QA * 2  # Increase the beam search width
)

        QandA=[]
        for i in output:

        
            # output = self.tokenizer.decode(output, skip_special_tokens=True)
            
            QandA.append(self.tokenizer.decode(i, skip_special_tokens=True).split("<sep>"))
        
        # return {"answer": output[0], "question": output[1]}
        return QandA

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

        output = self.DistracterGen.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=True,        # Enable sampling
        top_k=50,              # Consider the top 50 tokens for each generation step
        top_p=0.95,            # Use nucleus sampling to ensure diversity
        temperature=0.7,       # Add randomness to generation
        max_length=64,         # Limit distracter length
        num_return_sequences=3  # Return 3 distracters
    )

        output = self.tokenizer.decode(output[0], skip_special_tokens=True).split("<sep>")
        return output

    def gen(self, context, num_QA):
        QandA_pairs = self.genQandA(context, num_QA)
        data = []
        for answer, question in QandA_pairs:
            distracters = self.distracter(answer, question, context)
            data.append({
                "question": question,
                "answer": answer,
                "distracters": distracters
            })
        return data
        
    def train_QA(self,dataloader,epochs,learning_rate):
        optimizer = Adam(self.QandAGenerator.parameters(), lr=learning_rate)
        self.QandAGenerator.train()
        if torch.cuda.is_available():
            device='cuda'
        else:
            device='cpu'

        for epoch in range(epochs):
            tqdm.write(f'Epoch {epoch+1}/{epochs}')
            epoch_loss = 0
            progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}', leave=False)
            self.QandAGenerator.train()

            for batch_idx, batch in enumerate(progress_bar):
                batch = batch
                optimizer.zero_grad()

                source_token, target_token,mask_attention = batch
                output = self.QandAGenerator(input_ids=source_token.to(device), labels=target_token.to(device),attention_mask=mask_attention.to(device))
                loss = output.loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                progress_bar.set_postfix({
                    'Batch': batch_idx + 1,
                    'Loss': loss.item(),
                    'Avg Loss': epoch_loss / (batch_idx + 1)
                })

            tqdm.write(f'Epoch {epoch+1} completed with avg loss: {epoch_loss/len(dataloader)}')

    def train_dis(self):
        pass