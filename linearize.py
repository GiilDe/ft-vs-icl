import abc
import os

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers
from fairseq.models.transformer_lm import TransformerLanguageModel

from utils import DotDict


class LinearizedModel(nn.Module):
    """Creates a linearized version of a nn.Module.

    The linearized version of a model is a proper PyTorch model and can be
    trained as any other nn.Module.

    Args:
        model (nn.Module): The model to linearize. The trainable parameters of
            the linearized model will be initialized to the parameters of this
            model.
        init_model (nn.Module): A model of the same type as `model` containing
            the parameters around which the model is initialized. If not
            provided, `model` is used as the initialization model.
    """

    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        """Initializes the linearized model."""
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__
        for i, buffer in enumerate(buffers0):
            name = f"buffer{i}"
            self.register_buffer(name, buffer)

        self.func0 = lambda params, **kwargs: func0(params, self.buffers(), **kwargs)
        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

    def __call__(self, **kwargs) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, **kwargs),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp

class LinearizedTLM(TransformerLanguageModel):
    def __init__(
        self, model: TransformerLanguageModel, init_model: TransformerLanguageModel = None
    ):
        super().__init__(model.decoder)
        for p in self.parameters():
            p.requires_grad = False # don't need to train these as we will train the parameters in the linearized gpt's

        original_forward = model.decoder.forward
        decoder_extract_features_x = model.decoder
        decoder_extract_features_x.forward = lambda **kwargs: decoder_extract_features_x.extract_features_scriptable(**kwargs)[0]
        self.linearized_decoder_extract_features_x = LinearizedModel(model=decoder_extract_features_x, init_model=decoder_extract_features_x)

        decoder_extract_features_extra = model.decoder
        decoder_extract_features_extra.forward = lambda **kwargs: decoder_extract_features_extra.extract_features_scriptable(**kwargs)[1]
        self.linearized_decoder_extract_features_extra = LinearizedModel(model=decoder_extract_features_extra, init_model=decoder_extract_features_extra)

        decoder_output_layer = model.decoder
        decoder_output_layer.forward = decoder_output_layer.output_layer
        self.linearized_decoder_output_layer = LinearizedModel(model=decoder_output_layer, init_model=decoder_output_layer)

        model.decoder.extract_features_scriptable = lambda **kwargs: (self.linearized_decoder_extract_features_x.__call__(**kwargs), self.linearized_decoder_extract_features_extra.__call__(**kwargs))
        model.decoder.output_layer = self.linearized_decoder_output_layer.__call__

        model.decoder.forward = original_forward
