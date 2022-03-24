# embedding_constraint_callback

Embeddings are transformations that map elements from a finite vocabulary into positions in a low-dimensional space, and in [PyTorch](https://pytorch.org/) are implemented via the `Embedding` component. Training an `Embedding` adjusts the mappings for all the vocabulary elements, but in some situations it may be desirable to use predetermined mappings for some elements in the vocabulary. The `embedding_constraint_callback` repository provides a simple-to-use mechanism to achieve such an effect.

A specific use case for this type of constraint is handling of an expanding vocabulary. Consider an embedding and an associated classifier that uses a certain vocabulary. Suppose that, over time, there arises a need to support a vocabulary with more elements. Such support could be achieved by replacing the existing `Embedding` component by a newly-trained `Embedding` that supports a larger vocabulary, but provides exactly the same mappings for all previous vocabulary elements.

The mechanism is implemented as a component, called `EmbeddingConstraint`, that can be used as a 'callback' with the [pytorch-lightning](https://www.pytorchlightning.ai/) framework.


## Installation

The `EmbeddingConstraint` object is defined in a single file located at [callbacks/embedding_constraint.py](/callbacks/embedding_constraint.py). To use this component, simply copy the contents of this file into a project. 


## How to use

Consider the following minimal example for using pytorch-lightning to train a model. Suppose that a class `MyLightningModule` extends `LightningModule` and includes an `Embedding` component that maps a vocabulary of size `vocab_dim` into a space of dimension `embedding_dim` . Also consider that `train_dataloader` and `val_dataloader` are `DataLoader` objects that provide data for training and validation. Then a canonical use of a pytorch-lightning `Trainer` would be as follows.

```
model = MyLightningModule(vocab_dim, embedding_dim)
trainer = Trainer()
trainer.fit(model, train_dataloader, val_dataloader)
```

Suppose that a new vocabulary of size `vocab_dim_2` becomes available. Suppose that a new model should be trained to maintain the same mappings as in the first model for the first `vocab_dim` elements, but also support mappings for the new items in the vocabulary. The constrained training would be achieved by providing an `EmbeddingConstraint` callback to the trainer.

```
model_2 = MyLightningModule(vocab_dim_2, embedding_dim)
constraint = EmbeddingConstraint(model_2.embedding, model.embedding)
trainer_2 = Trainer(callbacks=[constraint])
trainer_2.fit(model_2, train_dataloader_2, val_dataloader_2)
```

As trainer fits the model, it adjusts the model parameters, including the mappings within the `model_2.embedding` component. After each training batch, however, the callback object adjusts the mappings to satisfy the constraint. Thus, throughout the training process, the mappings for the first `vocab_dim` elements correspond to the mappings provided by the first model. 


## Examples

The [examples](/examples/) folder includes a notebook with a complete working example ([/examples/example_produce.ipynb](examples/example_produce.ipynb)). 

