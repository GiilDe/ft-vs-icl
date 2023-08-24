import abc
import copy
import os

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers
from fairseq.models.transformer_lm import TransformerLanguageModel
from torch import autocast

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

    parmas0_sign = "<0>"

    def __init__(
        self,
        model: nn.Module,
        init_model: nn.Module = None,
        sum_extra_jvp_results: bool = True,
    ) -> None:
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
            param_original_name_processed = param_original_name.replace(
                ".", "_"
            )  # param names can't have dots
            self.register_parameter(
                param_original_name_processed + self.parmas0_sign, param0
            )
            self.register_parameter(param_original_name_processed, param1)

        for buffer, model_named_buffer in zip(buffers0, model.named_buffers()):
            buffer_original_name, buffer_original_model = model_named_buffer
            buffer_original_name_processed = buffer_original_name.replace(
                ".", "_"
            )  # buffer names can't have dots
            assert torch.equal(buffer, buffer_original_model)  # sanity check
            self.register_buffer(buffer_original_name_processed, buffer)

        self.func0 = lambda params, x: func0(params, self.buffers(), x)
        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

        self.sum_extra_jvp_results = sum_extra_jvp_results

    @autocast("cuda")
    def __call__(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(list(self.params0)),),
            (tuple(dparams),),
        )
        x = out[0] + dp[0]
        if self.sum_extra_jvp_results:
            extra = {}
            if "attn" in out[1]:
                extra["attn"] = [
                    out[1]["attn"][i] + dp[1]["attn"][i]
                    for i in range(len(out[1]["attn"]))
                ]
            extra["inner_states"] = [
                out[1]["inner_states"][i] + dp[1]["inner_states"][i]
                for i in range(len(out[1]["inner_states"]))
            ]
            extra["qkv_val"] = []
            for qkv_val_out, qkv_val_dp in zip(out[1]["qkv_val"], dp[1]["qkv_val"]):
                qkv_val_sum = {
                    key: qkv_val_out[key] + qkv_val_dp[key]
                    for key in qkv_val_out.keys()
                }
                extra["qkv_val"].append(qkv_val_sum)
            extra["self_attn_out_hiddens"] = [
                out[1]["self_attn_out_hiddens"][i] + dp[1]["self_attn_out_hiddens"][i]
                for i in range(len(out[1]["self_attn_out_hiddens"]))
            ]
        else:
            extra = out[1]
        if "attn" not in extra:
            extra["attn"] = [None]

        return x, extra


class LinearizedTLM(TransformerLanguageModel):
    def named_parameters(self, recurse=True):
        return self.linearized_model.named_parameters(recurse=True)

    def named_parameters_for_setting_grad(self):
        return [
            name_param
            for name_param in self.named_parameters()
            if self.linearized_model.parmas0_sign not in name_param[0]
        ]

    def named_buffers(self, recurse=True):
        return self.linearized_model.named_buffers(recurse=True)

    def __init__(
        self,
        model: TransformerLanguageModel,
        init_model: TransformerLanguageModel = None,
        sum_extra_jvp_results: bool = True,
    ):
        model_copy = copy.deepcopy(model)
        for p in model.parameters():
            p.requires_grad = False
            p.data = torch.empty(0)
        super().__init__(model.decoder)
        self.linearized_model = LinearizedModel(
            model=model_copy,
            init_model=model_copy,
            sum_extra_jvp_results=sum_extra_jvp_results,
        )
        del model_copy

        assert len(list(self.named_parameters())) / 2 == len(
            list(model.named_parameters())
        )

        assert len(list(self.named_parameters_for_setting_grad())) == len(
            list(model.named_parameters())
        )

        assert len(list(self.named_buffers())) == len(
            list(model.buffers())
        )  # the linearized model should have the same number of buffers as the original model. Self also contains the model.decoder buffers (redundant and never used) hence comparing with self.linearized_model

    def forward(self, x):
        # use the taylorized version of the model.
        return self.linearized_model(x)

    def __call__(self, x):
        return self.forward(x)
