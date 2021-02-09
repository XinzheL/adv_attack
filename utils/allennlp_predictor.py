from allennlp.predictors import Predictor
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