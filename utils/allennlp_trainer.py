# for training 
from allennlp.training.trainer import GradientDescentTrainer
from allennlp.training import TensorBoardCallback
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file
import torch.optim as optim
from allennlp.training.trainer import Trainer
from transformers import AdamW
from allennlp.training.optimizers import AdamOptimizer
import allennlp
from typing import Tuple
from allennlp.data.data_loaders import SimpleDataLoader
from .allennlp_data import MySimpleDataLoader
from .allennlp_transformers import MyAttentionLayer, MyPooler

from torch import save, load
import torch
from typing import Dict, Union, Optional, List
from copy import deepcopy
import os


def train_sst_model(output_dir, train_data, dev_data, MODEL_TYPE, \
    EMBEDDING_TYPE = None, 
    num_epochs=3,
    bsz = 32,
    TRAIN_TYPE=None,
    LABELS = [0, 1],
    activation=None):

    print('Begin Training...')
    # initialize vocab with specified 'labels' namespace
    from allennlp.data.fields import LabelField
    tmp_instances = []
    for label in LABELS:
        # _label_index should not be set, otherwise Vocabulary couter would not count LabelField
        tmp_instances.append(Instance(fields={'labels': LabelField(str(label), skip_indexing=False)}))
    vocab = Vocabulary.from_instances(tmp_instances) 
    
    if "bert" in MODEL_TYPE:

        
        token_embedding = PretrainedTransformerEmbedder(model_name=MODEL_TYPE, train_parameters=True)

        # 3. seq2vec encoder
        encoder = MyPooler(pretrained_model=MODEL_TYPE, dropout=0.1, requires_grad=True)

        # 4. construct model
        
        model = BertForClassification(vocab, BasicTextFieldEmbedder({"tokens": token_embedding}), encoder)
        
    else:
        # 2. token embedding
        vocab.extend_from_instances(train_data)
        # vocab = Vocabulary.from_instances(train_data)
        vocab_size = vocab.get_vocab_size('tokens')
        word_embedding_dim = 300
        if EMBEDDING_TYPE is None:
            token_embedding = Embedding(num_embeddings=vocab_size, embedding_dim=word_embedding_dim)
        elif EMBEDDING_TYPE == "w2v":
            
            embedding_path = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
            weight = _read_pretrained_embeddings_file(embedding_path,
                                                    embedding_dim=word_embedding_dim,
                                                    vocab=vocab,
                                                    namespace="tokens")
            token_embedding = Embedding(num_embeddings=vocab_size,
                                        embedding_dim=word_embedding_dim,
                                        weight=weight,
                                        trainable=False)
        
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
         
        # 3. seq2vec encoder
        if 'lstm' in MODEL_TYPE:
            encoder = PytorchSeq2VecWrapper(LSTM(word_embedding_dim,
                                                        hidden_size=512,
                                                        num_layers=2,
                                                        batch_first=True))
        elif 'cnn' in MODEL_TYPE:
            encoder = CnnEncoder(word_embedding_dim, num_filters=6, conv_layer_activation=activation)
        # elif 'transformer' in MODEL_TYPE:
        #     
        #     # TODO: try different scoring function
        #     encoder = TransformerLayer(hidden_size=word_embedding_dim, intermediate_size=521, \
        #         num_attention_heads=5, attention_dropout=0.0, hidden_dropout=0.0, \
        #         activation="relu", add_cross_attention=False)
        else:
            print(f'Invalid MODEL_TYPE {MODEL_TYPE}')
            exit()

        # 4. construct model
        if "dot_product" in MODEL_TYPE:
            scoring_func = "dot_product"
            
            attention = MyAttentionLayer(
                hidden_size=word_embedding_dim,
                num_attention_heads=5,
                attention_dropout=0,
                hidden_dropout=0,
                scoring_func= scoring_func,
            )
        else:
            attention = None
        model = SSTClassifier(word_embeddings, encoder, vocab, attention=attention)
    model.cuda()
      
    # 5. train model from scratch and save its weights
    # define optim
    if "bert" in MODEL_TYPE:
        optimizer = AdamW(model.parameters(),
                        lr=5e-5,    # Default learning rate
                        eps=1e-8    # Default epsilon value
                        )
    else:
        
        parameters = [[n, p] for n, p in model.named_parameters() if p.requires_grad]
        optimizer = AdamOptimizer(parameters)
    # from allennlp.training.learning_rate_schedulers import LearningRateScheduler
    # scheduler = LearningRateScheduler()
    if TRAIN_TYPE is None or TRAIN_TYPE == "error_max":
        train_loader, dev_loader = build_data_loaders(list(train_data), list(dev_data), vocab, bsz=bsz)
        trainer = build_trainer(model, output_dir, train_loader, dev_loader, num_epochs=num_epochs, optimizer=optimizer)
    elif TRAIN_TYPE == "error_min":
        if 'bert' in MODEL_TYPE:
            vocab_namespace='tags'
        elif 'lstm' in MODEL_TYPE:
            vocab_namespace='tokens'
        train_loader = MySimpleDataLoader(train_data, batch_size=bsz, shuffle=False, vocab=vocab, )
        dev_loader = SimpleDataLoader(dev_data, batch_size=bsz, shuffle=False, vocab=vocab)
        trainer = build_error_min_unlearnable_trainer(model, output_dir, train_loader,\
            vocab, vocab_namespace, \
            dev_loader, num_epochs=num_epochs, optimizer=optimizer, )
    
    trainer.train()

    # save vocab, considering model would be saved automatically, I think 
    # somehome, vocab should also be saved somewhere in allennlp process
    vocab.save_to_files(output_dir + "vocab")
    
    

# The other `build_*` methods are things we've seen before, so they are
# in the setup section above.
def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
    vocab: Vocabulary,
    bsz: int = 64
) -> Tuple[allennlp.data.DataLoader, allennlp.data.DataLoader]:
    # Note that DataLoader is imported from allennlp above, *not* torch.
    # We need to get the allennlp-specific collate function, which is
    # what actually does indexing and batching.
    train_loader = SimpleDataLoader(train_data, batch_size=bsz, shuffle=True, vocab=vocab)
    dev_loader = SimpleDataLoader(dev_data, batch_size=bsz, shuffle=False, vocab=vocab)
    return train_loader, dev_loader

def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: allennlp.data.DataLoader,
    dev_loader: allennlp.data.DataLoader,
    num_epochs: int = 3,
    optimizer=None
    
) -> Trainer:
    
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        learning_rate_scheduler=None,
        patience=3,
        callbacks=[TensorBoardCallback(serialization_dir)]
    )
    return trainer

def build_error_min_unlearnable_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: allennlp.data.DataLoader,
    vocab,
    vocab_namespace,
    dev_loader: allennlp.data.DataLoader,
    num_epochs: int = 3,
    optimizer=None,
    
    
) -> Trainer:
    


    trainer = ErrorMinUnlearnableTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        vocab=vocab,
        vocab_namespace=vocab_namespace,
        validation_data_loader=dev_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        learning_rate_scheduler=None,
        patience=3,
        callbacks=[TensorBoardCallback(serialization_dir)]
    )
    return trainer



from allennlp.training.trainer import GradientDescentTrainer, TrainerCallback
import datetime
import logging
import math
import os
import re
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from copy import deepcopy
import allennlp.nn.util as nn_util

from allennlp.common.util import int_to_device

import torch
import torch.distributed as dist
from torch.cuda import amp
import torch.optim.lr_scheduler
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_

from allennlp.common import Lazy, Registrable, Tqdm
from allennlp.data.tokenizers import Token
from allennlp.common import util as common_util
from allennlp.common.checks import ConfigurationError, check_for_gpu
from allennlp.data import DataLoader, TensorDict
from allennlp.models.model import Model
from allennlp.training import util as training_util
from allennlp.training.checkpointer import Checkpointer
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.metric_tracker import MetricTracker
from allennlp.training.momentum_schedulers import MomentumScheduler
from allennlp.training.moving_average import MovingAverage
from allennlp.training.optimizers import Optimizer
from allennlp.training.tensorboard_writer import TensorBoardWriter

from .universal_attack import UniversalAttack
from utils.allennlp_predictor import AttackPredictorForBiClassification


from allennlp.data.data_loaders.data_loader import allennlp_collate
from allennlp.data.dataset_readers import DatasetReader

logger = logging.getLogger(__name__)

class ErrorMinUnlearnableTrainer(GradientDescentTrainer):
    def __init__(
        self,
        model: Model,
        optimizer: torch.optim.Optimizer,
        data_loader: DataLoader,
        vocab,
        vocab_namespace,
        patience: Optional[int] = None,
        validation_metric: Union[str, List[str]] = "-loss",
        validation_data_loader: DataLoader = None,
        num_epochs: int = 20,
        serialization_dir: Optional[str] = None,
        checkpointer: Checkpointer = None,
        cuda_device: Optional[Union[int, torch.device]] = None,
        grad_norm: Optional[float] = None,
        grad_clipping: Optional[float] = None,
        learning_rate_scheduler: Optional[LearningRateScheduler] = None,
        momentum_scheduler: Optional[MomentumScheduler] = None,
        moving_average: Optional[MovingAverage] = None,
        callbacks: List[TrainerCallback] = None,
        distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1,
        num_gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        num_trigger_tokens: int = 3,
       
    ) -> None:
        super(ErrorMinUnlearnableTrainer, self).__init__(model, optimizer, data_loader, patience, \
                        validation_metric, validation_data_loader, num_epochs, serialization_dir,\
                        checkpointer, cuda_device, grad_norm, grad_clipping, learning_rate_scheduler, \
                        momentum_scheduler, moving_average, callbacks)

        #     train_data_all_classes = {}
        
        # initialize trigger tokens for each class
        trigger_tokens = []
        for _ in range(num_trigger_tokens):
            trigger_tokens.append(Token("the"))
        
        self.trigger_tokens_dict = {}
        self.vocab = vocab
        self.vocab_namespace = vocab_namespace
        self.label_ids = [0, 1]
        for k in self.label_ids:
            self.trigger_tokens_dict[k] = deepcopy(trigger_tokens)


        



    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Trains one epoch and returns metrics.
        """
        logger.info("Epoch %d/%d", epoch, self._num_epochs - 1)
        cpu_memory_usage = []
        for worker, memory in common_util.peak_cpu_memory().items():
            cpu_memory_usage.append((worker, memory))
            logger.info(f"Worker {worker} memory usage: {common_util.format_size(memory)}")
        gpu_memory_usage = []
        for gpu, memory in common_util.peak_gpu_memory().items():
            gpu_memory_usage.append((gpu, memory))
            logger.info(f"GPU {gpu} memory usage: {common_util.format_size(memory)}")

        regularization_penalty = self.model.get_regularization_penalty()

        train_loss = 0.0
        batch_loss = 0.0
        train_reg_loss = None if regularization_penalty is None else 0.0
        batch_reg_loss = None if regularization_penalty is None else 0.0

        # Set the model to "train" mode.
        self._pytorch_model.train()

        
        
        # Get tqdm for the training batches
        batch_generator = iter(self.data_loader)
        batch_group_generator = common_util.lazy_groups_of(
            batch_generator, self._num_gradient_accumulation_steps
        )

        logger.info("Training")

        num_training_batches: Union[int, float]
        try:
            len_data_loader = len(self.data_loader)
            num_training_batches = math.ceil(
                len_data_loader / self._num_gradient_accumulation_steps
            )
        except TypeError:
            num_training_batches = float("inf")

        # Having multiple tqdm bars in case of distributed training will be a mess. Hence only the primary's
        # progress is shown
        if self._primary:
            batch_group_generator_tqdm = Tqdm.tqdm(
                batch_group_generator, total=num_training_batches
            )
        else:
            batch_group_generator_tqdm = batch_group_generator

        self._last_log = time.time()

        batches_this_epoch = 0
        if self._batch_num_total is None:
            self._batch_num_total = 0

        done_early = False
        for batch_group in batch_group_generator_tqdm:
            if self._distributed:
                # Check whether the other workers have stopped already (due to differing amounts of
                # data in each). If so, we can't proceed because we would hang when we hit the
                # barrier implicit in Model.forward. We use a IntTensor instead a BoolTensor
                # here because NCCL process groups apparently don't support BoolTensor.
                done = torch.tensor(0, device=self.cuda_device)
                torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
                if done.item() > 0:
                    done_early = True
                    logger.warning(
                        f"Worker {torch.distributed.get_rank()} finishing training early! "
                        "This implies that there is an imbalance in your training "
                        "data across the workers and that some amount of it will be "
                        "ignored. A small amount of this is fine, but a major imbalance "
                        "should be avoided. Note: This warning will appear unless your "
                        "data is perfectly balanced."
                    )
                    break

            batches_this_epoch += 1
            self._batch_num_total += 1
            batch_num_total = self._batch_num_total

            # Zero gradients.
            # NOTE: this is actually more efficient than calling `self.optimizer.zero_grad()`
            # because it avoids a read op when the gradients are first updated below.
            for param_group in self.optimizer.param_groups:
                for p in param_group["params"]:
                    p.grad = None

            batch_loss = 0.0
            batch_group_outputs = []

            noisy_batch = []
            for batch in batch_group:

                ##### prepend trigger tokens
                batch_dict = {}
                for label in self.label_ids:
                    noisy_batch.append(UniversalAttack.prepend_batch(
                            UniversalAttack.filter_instances( \
                                batch, label_filter=label, vocab=self.vocab
                            ), \
                            trigger_tokens=self.trigger_tokens_dict[label], \
                            vocab = self.vocab
                        )
                    ) 
                    batch_dict[label] = UniversalAttack.filter_instances( batch, label_filter=label, vocab=self.vocab)

                for instance in batch:
                    instance.index_fields(self.vocab)
                batch = allennlp_collate(batch)
                if self.cuda_device is not None:
                    batch = nn_util.move_to_device(batch, self.cuda_device)
                    
                #####

                with amp.autocast(self._use_amp):
                    batch_outputs = self.batch_outputs(batch, for_training=True)
                    batch_group_outputs.append(batch_outputs)
                    loss = batch_outputs["loss"]
                    reg_loss = batch_outputs.get("reg_loss")
                    if torch.isnan(loss):
                        raise ValueError("nan loss encountered")
                    loss = loss / len(batch_group)

                    batch_loss += loss.item()
                    if reg_loss is not None:
                        reg_loss = reg_loss / len(batch_group)
                        batch_reg_loss = reg_loss.item()
                        train_reg_loss += batch_reg_loss  # type: ignore

                if self._scaler is not None:
                    self._scaler.scale(loss).backward()
                else:
                    loss.backward()

                ##### Update Trigger Tokens
                # predictor
                predictor = AttackPredictorForBiClassification(self._pytorch_model, DatasetReader())
                universal = UniversalAttack(predictor)
                for label in self.label_ids:
                    if len(batch_dict[label]) > 0:
                        self.trigger_tokens_dict[label] = universal.update_triggers( batch_dict[label],\
                            predictor, self.vocab,  \
                            self.trigger_tokens_dict[label], sign=1, vocab_namespace=self.vocab_namespace)

                #####

            train_loss += batch_loss

            batch_grad_norm = self.rescale_gradients()

            # This does nothing if batch_num_total is None or you are using a
            # scheduler which doesn't update per batch.
            if self._learning_rate_scheduler:
                self._learning_rate_scheduler.step_batch(batch_num_total)
            if self._momentum_scheduler:
                self._momentum_scheduler.step_batch(batch_num_total)

            if self._scaler is not None:
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()

            # Update moving averages
            if self._moving_average is not None:
                self._moving_average.apply(batch_num_total)

            # Update the description with the latest metrics
            metrics = training_util.get_metrics(
                self.model,
                train_loss,
                train_reg_loss,
                batch_loss,
                batch_reg_loss,
                batches_this_epoch,
                world_size=self._world_size,
                cuda_device=self.cuda_device,
            )

            if self._primary:
                # Updating tqdm only for the primary as the trainers wouldn't have one
                description = training_util.description_from_metrics(metrics)
                batch_group_generator_tqdm.set_description(description, refresh=False)

                if self._checkpointer is not None:
                    self._checkpointer.maybe_save_checkpoint(self, epoch, batches_this_epoch)

            for callback in self._callbacks:
                callback.on_batch(
                    self,
                    batch_group,
                    batch_group_outputs,
                    metrics,
                    epoch,
                    batches_this_epoch,
                    is_training=True,
                    is_primary=self._primary,
                    batch_grad_norm=batch_grad_norm,
                )

        if self._distributed and not done_early:
            logger.warning(
                f"Worker {torch.distributed.get_rank()} completed its entire epoch (training)."
            )
            # Indicate that we're done so that any workers that have remaining data stop the epoch early.
            done = torch.tensor(1, device=self.cuda_device)
            torch.distributed.all_reduce(done, torch.distributed.ReduceOp.SUM)
            assert done.item()

        # Let all workers finish their epoch before computing
        # the final statistics for the epoch.
        if self._distributed:
            dist.barrier()

        metrics = training_util.get_metrics(
            self.model,
            train_loss,
            train_reg_loss,
            batch_loss=None,
            batch_reg_loss=None,
            num_batches=batches_this_epoch,
            reset=True,
            world_size=self._world_size,
            cuda_device=self.cuda_device,
        )

        for (worker, memory) in cpu_memory_usage:
            metrics["worker_" + str(worker) + "_memory_MB"] = memory / (1024 * 1024)
        for (gpu_num, memory) in gpu_memory_usage:
            metrics["gpu_" + str(gpu_num) + "_memory_MB"] = memory / (1024 * 1024)
        return metrics








