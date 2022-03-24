import unittest
import logging
import torch
import torch.nn as nn
import pytorch_lightning as pl
from os import cpu_count
from torch.utils.data import Dataset, DataLoader
from callbacks.embedding_constraint import EmbeddingConstraint

# use all cpus for tests - avoid warnings about performance in DataLoader
n_cpus = cpu_count()

# set logging verbosity to disable information from pytorch lightning
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class ExampleDataset(Dataset):
    """dataset that presents integers with a modulus

    ** for unit tests only **
    """

    def __init__(self, n: int):
        """
        :param n: size of dataset
        """
        super().__init__()
        self.data = torch.arange(0, n)
        self.label = self.data.add(1)
        self.data = self.data.int()
        self.label[-1] = 0

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]


class ExampleModule(pl.LightningModule):
    """module that trains an embedding

    ** for unit tests only **
    """

    def __init__(self, vocab_size, embedding_dim, lr=1e-3):
        """
        :param vocab_size: number of elements in the vocabulary
        :param embedding_dim: number of embedding dimensions
        :param lr: base learning rate
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        :param x: tensor with batches of integers
        :return: prediction of one element in vocab_size space
        """
        batch_size = x.shape[0]
        x = self.embedding(x.int())
        x = x.view(batch_size, self.hparams.embedding_dim)
        return self.output(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        data, labels = batch
        pred = self.forward(data)
        loss = self.criterion(pred, labels)
        return loss


class VocabExpansionTestCase(unittest.TestCase):
    """tests training embedding with a constraint callback"""

    def test_example_dataset(self):
        """check that the example dataset produces tensors"""
        dataset = ExampleDataset(8)
        result = next(iter(dataset))
        self.assertEqual(result[0].numel(), 1)
        self.assertEqual(result[1].numel(), 1)

    def test_example_loader(self):
        """check data loader creates batches of data"""
        dataset = ExampleDataset(8)
        loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)
        result = [_ for _ in loader]
        self.assertEqual(len(result), 2)
        batch = result[0]
        self.assertListEqual(list(batch[0].shape), [4])
        self.assertListEqual(list(batch[1].shape), [4])

    def test_train_embedding(self):
        """check base interactions between dataset, model, and trainer"""
        dataset = ExampleDataset(8)
        loader = DataLoader(dataset, batch_size=5, shuffle=True,
                            num_workers=n_cpus)
        trainer = pl.Trainer(logger=False,
                             enable_checkpointing=False,
                             gpus=0,
                             max_epochs=4,
                             callbacks=[],
                             enable_model_summary=False,
                             enable_progress_bar=False,
                             log_every_n_steps=1)
        model = ExampleModule(8, 2)
        trainer.fit(model, loader)
        weights = model.embedding.weight.detach()
        self.assertListEqual(list(weights.shape), [8, 2])

    def test_raise_misconfiguration_with_matrix(self):
        """attempt to define a constraint with a matrix"""
        embedding = nn.Embedding(6, 2)
        weight = embedding.weight.detach().numpy()
        with self.assertRaises(Exception):
            EmbeddingConstraint(embedding, weight)

    def test_raise_misconfiguration(self):
        """attempt to define a constraint with a list"""
        embedding = nn.Embedding(6, 2)
        with self.assertRaises(Exception):
            EmbeddingConstraint(embedding, [])

    def test_train_embedding_constraint(self):
        """train a simple neural network model"""
        dataset = ExampleDataset(8)
        loader = DataLoader(dataset, batch_size=5, shuffle=True,
                            num_workers=n_cpus)
        model = ExampleModule(8, 2)
        # create an initial embedding that will act as a constraint
        fixed_embedding = nn.Embedding(6, 2)
        embedding_constraint = EmbeddingConstraint(model.embedding,
                                                   fixed_embedding)
        fixed_weight = fixed_embedding.weight.detach()
        # train the model with the constraint
        trainer = pl.Trainer(logger=False,
                             enable_checkpointing=False,
                             gpus=0,
                             max_epochs=4,
                             callbacks=[embedding_constraint],
                             enable_model_summary=False,
                             enable_progress_bar=False,
                             log_every_n_steps=1)
        trainer.fit(model, loader)
        # the fitted embedding should have dimensions 8x2
        result = model.embedding.weight.detach()
        self.assertListEqual(list(result.shape), [8, 2])
        # fitted embedding should have exactly the same weights as
        # the fixed_embedding object
        self.assertListEqual(list(result.numpy()[:6, 0]),
                             list(fixed_weight.numpy()[:6, 0]))
        self.assertListEqual(list(result.numpy()[:6, 1]),
                             list(fixed_weight.numpy()[:6, 1]))



