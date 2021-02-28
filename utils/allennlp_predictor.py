from allennlp.predictors import Predictor
import torch
class AttackPredictorForBiClassification(Predictor):
    def _json_to_instance(self, json_dict):
        from allennlp.data import Instance
        from copy import deepcopy
        fields = deepcopy(json_dict) # ensure returned `fields` of `Instance` has different memory id from `json_dict`
        return Instance(fields) 

    def predictions_to_labeled_instances(self, instance, output_dict):
        from allennlp.data.fields import LabelField
        label_id = int(output_dict['logits'].argmax()) # this could be used as _label_id for `LabelField`
        instance.fields['predict_label'] = LabelField(label_id, skip_indexing=True) 
        return [instance]

    def get_interpretable_layer(self) -> torch.nn.Module:
        from transformers.models.gpt2.modeling_gpt2 import GPT2Model
        from transformers.models.bert.modeling_bert import BertEmbeddings
        from transformers.models.albert.modeling_albert import AlbertEmbeddings
        from transformers.models.roberta.modeling_roberta import RobertaEmbeddings
        from allennlp.modules.text_field_embedders.text_field_embedder import TextFieldEmbedder
        from allennlp.modules.text_field_embedders.basic_text_field_embedder import (
            BasicTextFieldEmbedder,
        )
        from allennlp.modules.token_embedders.embedding import Embedding
        from transformers.models.distilbert import modeling_distilbert

        for module in self._model.modules():
            if isinstance(module, BertEmbeddings):
                return module.word_embeddings
            if isinstance(module, RobertaEmbeddings):
                return module.word_embeddings
            if isinstance(module, AlbertEmbeddings):
                return module.word_embeddings
            if isinstance(module, GPT2Model):
                return module.wte
            if isinstance(module, modeling_distilbert.Embeddings):
                return module.word_embeddings

        for module in self._model.modules():
            if isinstance(module, TextFieldEmbedder):

                if isinstance(module, BasicTextFieldEmbedder):
                    # We'll have a check for single Embedding cases, because we can be more efficient
                    # in cases like this.  If this check fails, then for something like hotflip we need
                    # to actually run the text field embedder and construct a vector for each token.
                    if len(module._token_embedders) == 1:
                        embedder = list(module._token_embedders.values())[0]
                        if isinstance(embedder, Embedding):
                            if embedder._projection is None:
                                # If there's a projection inside the Embedding, then we need to return
                                # the whole TextFieldEmbedder, because there's more computation that
                                # needs to be done than just multiply by an embedding matrix.
                                return embedder
                return module
        raise RuntimeError("No embedding module found!")
