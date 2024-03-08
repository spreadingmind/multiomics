import torch
import numpy as np
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import Trainer, TrainingArguments
from datasets import Dataset

class Autoencoder(nn.Module):
    def __init__(self, input_dims, n_out):
        super(Autoencoder, self).__init__()

        self.encoder = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5)
        ) for dim in input_dims])

        self.bottleneck = nn.Linear(128 * len(input_dims), n_out)
        self.decoder = nn.ModuleList([nn.Sequential(
            nn.Linear(n_out, dim),
            nn.Sigmoid()
        ) for dim in input_dims])

    def forward(self, inputs):
        encoded = [encoder(input) for encoder, input in zip(self.encoder, inputs)]
        merged = torch.cat(encoded, dim=1)
        bottleneck = self.bottleneck(merged)
        decoded = [decoder(bottleneck) for decoder in self.decoder]
        return decoded

    def encode(self, inputs):
        encoded = [encoder(input) for encoder, input in zip(self.encoder, inputs)]
        merged = torch.cat(encoded, dim=1)
        bottleneck = self.bottleneck(merged)
        return bottleneck


class EncoderPipeline():
    def __init__(self, modalities, n_out):
        self.modalities = modalities
        self.input_dims = [mod.shape[1] for mod in self.modalities]
        self.model = Autoencoder(self.input_dims, n_out)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dna_train, rna_train, methylation_train = [
            torch.tensor(mod, dtype=torch.float32) for mod in self.modalities
        ]
        train_dataset = TensorDataset(dna_train, rna_train, methylation_train)
        self.train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        self.train_dataset = Dataset.from_dict({
            "inputs": train_dataset.tensors,
        })


    def train(self):
        self.model.to(self.device)

        # Optimizer and Loss Function
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training Loop
        for epoch in range(50):
            self.model.train()
            running_loss = 0.0
            for data in self.train_loader:
                inputs = [modality.to(self.device) for modality in data]
                targets = inputs

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = sum(criterion(output, target)
                           for output, target in zip(outputs, targets))
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            print(f"Epoch {epoch+1}, Loss: {running_loss}")

    # def train(self):
    #     self.model.to(self.device)

    #     training_args = TrainingArguments(
    #         output_dir='data/transformers-train/results',
    #         num_train_epochs=50,
    #         learning_rate=0.001,
    #         per_device_train_batch_size=32,
    #         per_device_eval_batch_size=32,
    #         warmup_steps=100,
    #         weight_decay=0.01,
    #         logging_dir='data/transformers-train/results/logs',
    #         logging_steps=10,
    #     )
    #     trainer = Trainer(
    #         model=self.model,
    #         args=training_args,
    #         train_dataset=self.train_dataset,
    #     )

    #     trainer.train()        

    def encode(self):
        self.model.eval()
        output = []
        with torch.no_grad():
            for data in self.train_loader:
                inputs = [modality.to(self.device) for modality in data]
                outputs = self.model.encode(inputs)
                output.append(outputs.cpu())
        output = np.vstack(output)
        return output