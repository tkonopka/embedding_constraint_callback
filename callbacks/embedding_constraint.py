# Copyright Tomasz Konopka
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Callback class that fixes a portion of an Embedding
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.base import Callback
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class EmbeddingConstraint(Callback):
    r"""
    Callback used to constrain an embedding during training
    """

    def __init__(self,
                 embedding: "nn.Embedding",
                 constraint):
        """set up a constraint on training of an embedding

        :param embedding: embedding component that is meant to be constrained
            during training
        :param constraint: object holding the constrain requirements. The
            constraint can be provided as a tensor or as an existing embedding
        """
        super().__init__()
        self.embedding = embedding
        self.constraint = None
        if type(constraint) is torch.Tensor:
            self.constraint = constraint.detach().clone()
        if type(constraint) is nn.Embedding:
            self.constraint = constraint.weight.detach().clone()
        if self.constraint is None:
            raise MisconfigurationException(f"'constraint' must be a tensor or an Embedding")

    def _apply_constraint(self):
        n = self.constraint.shape[0]
        requires_grad = self.embedding.weight.requires_grad
        self.embedding.weight.requires_grad = False
        self.embedding.weight[:n] = self.constraint
        self.embedding.weight.requires_grad = requires_grad

    def on_batch_end(self,
                     trainer: "pl.Trainer",
                     pl_module: "pl.LightningModule") -> None:
        """Called when the training batch ends."""
        self._apply_constraint()
