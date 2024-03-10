import torch
import numpy as np
import random
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from transformers import Trainer, TrainingArguments, set_seed
from datasets import Dataset
from torch.utils.data.dataloader import default_collate



class Autoencoder(nn.Module):
    def __init__(self, input_dims, n_out):
        super(Autoencoder, self).__init__()

        dna_encoder = nn.Sequential(
            nn.Linear(input_dims[0], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        rna_encoder = nn.Sequential(
            nn.Linear(input_dims[1], 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        met_encoder = nn.Sequential(
            nn.Linear(input_dims[2], 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.encoder = nn.ModuleList([dna_encoder, rna_encoder, met_encoder])
        self.bottleneck = nn.Linear(128 * len(input_dims), n_out)
        self.decoder = nn.ModuleList([nn.Sequential(
            nn.Linear(n_out, dim),
            nn.Sigmoid()
        ) for dim in input_dims])

    def forward(self, inputs):
        encoded = [encoder(input)
                   for encoder, input in zip(self.encoder, inputs)]
        merged = torch.cat(encoded, dim=1)
        bottleneck = self.bottleneck(merged)
        decoded = [decoder(bottleneck) for decoder in self.decoder]
        return decoded

    def encode(self, inputs):
        encoded = [encoder(input)
                   for encoder, input in zip(self.encoder, inputs)]
        merged = torch.cat(encoded, dim=1)
        bottleneck = self.bottleneck(merged)
        return bottleneck


class BaseAutoencoder(nn.Module):
    def __init__(self, input_dims, n_out):
        super(BaseAutoencoder, self).__init__()
        hidden_dim = sum([d for d in input_dims])
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Dropout(0.5)
        )
        self.bottleneck = nn.Linear(128, n_out)
        self.decoder = nn.Sequential(
            nn.Linear(n_out, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = torch.cat(inputs, dim=1)
        encoded = self.encoder(inputs)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        return decoded

    def encode(self, inputs):
        inputs = torch.cat(inputs, dim=1)
        encoded = self.encoder(inputs)
        bottleneck = self.bottleneck(encoded)
        return bottleneck


def custom_collate_fn(batch):
    modalities_separated = list(zip(*[item['inputs'] for item in batch]))
    batched_modalities = [default_collate(
        modality) for modality in modalities_separated]
    return {"inputs": batched_modalities}


class EncoderPipeline():
    def __init__(self, modalities, n_out, random_state, base=False):
        self.modalities = modalities
        self.random_state = random_state
        self.input_dims = [mod.shape[1] for mod in self.modalities]

        self.model = BaseAutoencoder(
            self.input_dims, n_out) if base else Autoencoder(self.input_dims, n_out)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        dna_train, rna_train, methylation_train = [
            torch.tensor(mod, dtype=torch.float32) for mod in self.modalities
        ]
        data = []
        for i in range(dna_train.shape[0]):
            sample = [dna_train[i], rna_train[i], methylation_train[i]]
            data.append(sample)
        self.train_dataset = Dataset.from_dict({'inputs': data})
        self.train_dataset.set_format(type='torch', columns=['inputs'])

        tensor_dataset = TensorDataset(dna_train, rna_train, methylation_train)
        self.train_loader = DataLoader(
            tensor_dataset, batch_size=32, shuffle=False)

    def train(self):
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        set_seed(self.random_state)

        self.model.to(self.device)

        training_args = TrainingArguments(
            output_dir='data/transformers-train/results-2',
            num_train_epochs=50,
            learning_rate=0.001,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='data/transformers-train/results/logs',
            logging_steps=10,
            save_total_limit=1,
            lr_scheduler_type='linear'
        )
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=custom_collate_fn
        )

        trainer.train()

    def encode(self):
        self.model.to(self.device)
        self.model.eval()
        output = []
        with torch.no_grad():
            for data in self.train_loader:
                inputs = [modality.to(self.device) for modality in data]
                outputs = self.model.encode(inputs)
                output.append(outputs.cpu())
        output = np.vstack(output)
        return output
