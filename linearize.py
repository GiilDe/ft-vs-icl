import abc
import os

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers
from fairseq.models.transformer_lm import TransformerLanguageModel


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

        # self.params = nn.ParameterList(params)
        # self.params0 = nn.ParameterList(params0)
        self.params = params
        self.params0 = params0
        self._model_name = model.__class__.__name__

        for param0, param1, model_named_param in zip(
            self.params0, self.params, model.named_parameters()
        ):
            param_original_name, param_original_model = model_named_param
            assert torch.equal(param0, param_original_model) and torch.equal(
                param1, param_original_model
            )  # sanity check
            param_original_name_processed = param_original_name.replace(".", "_") # param names can't have dots
            self.register_parameter(param_original_name_processed + "_0", param0)
            self.register_parameter(param_original_name_processed, param1)

        for buffer, model_named_buffer in zip(buffers0, model.named_buffers()):
            buffer_original_name, buffer_original_model = model_named_buffer
            buffer_original_name_processed = buffer_original_name.replace(".", "_") # buffer names can't have dots
            assert torch.equal(buffer, buffer_original_model)  # sanity check
            self.register_buffer(buffer_original_name_processed, buffer)

        # for i, buffer in enumerate(buffers0):
        #     name = f"buffer{i}"
        #     self.register_buffer(name, buffer)

        self.func0 = lambda params, x: func0(params, self.buffers(), x)
        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

    def __call__(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(list(self.params0)),),
            (tuple(dparams),),
        )
        x = out[0] + dp[0]
        extra = {key: out[1][key] + dp[1][key] for key in out[1].keys()}
        return x, extra


class LinearizedTLM(TransformerLanguageModel):
    def __init__(
        self,
        model: TransformerLanguageModel,
        init_model: TransformerLanguageModel = None,
    ):
        super().__init__(model.decoder)
        for p in self.parameters():
            p.requires_grad = False
        self.linearized_model = LinearizedModel(model=model, init_model=init_model)

        assert len(
            list(
                filter(
                    lambda named_param: named_param[1].requires_grad,
                    self.named_parameters(),
                )
            )
        ) == len(
            list(model.named_parameters())
        )  # the linearized model should have twice the number of parameters as the original model, but only half of them are trainable.

        assert len(list(self.linearized_model.buffers())) == len(
            list(model.buffers())
        )  # the linearized model should have the same number of buffers as the original model. Self also contains the model.decoder buffers (redundant and never used) hence comparing with self.linearized_model

    def forward(self, x):
        # use the taylorized version of the model.
        return self.linearized_model(x)

    def __call__(self, x):
        return self.forward(x)
