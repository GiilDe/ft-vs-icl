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
            lambda param: self.func0(param, x)[0],
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp

class LinearizedGPT(TransformerLanguageModel):
    def __init__(
        self, model: TransformerLanguageModel, init_model: TransformerLanguageModel = None
    ):
        super().__init__(model.decoder)
        for p in self.parameters():
            p.requires_grad = False # don't need to train these as we will train the parameters in the linearized model
        self.linearized_model = LinearizedModel(model=model, init_model=init_model)

    def forward(self, x):
        # use the taylorized version of the model.
        return self.linearized_model(x)

    def __call__(self, x):
        return self.forward(x), None

    # def max_positions(self):
    #     """Maximum length supported by the model."""
    #     return self.decoder.max_positions()

    # def set_num_updates(self, num_updates):
    #     """State from trainer to pass along to model at every update."""
    #     for m in self.modules():
    #         if hasattr(m, "set_num_updates") and m != self:
    #             m.set_num_updates(num_updates)


# class LinearizedImageEncoder(abc.ABC, nn.Module):
#     """Creates a linearized version of an image encoder."""

#     def __init__(
#         self, args=None, keep_lang=False, image_encoder=None, init_encoder=None
#     ):
#         super().__init__()
#         if image_encoder is None:
#             image_encoder = ImageEncoder(args, keep_lang)
#         if init_encoder is None:
#             init_encoder = image_encoder

#         # Copy the attributes from the image encoder.
#         self.train_preprocess = image_encoder.train_preprocess
#         self.val_preprocess = image_encoder.val_preprocess
#         self.cache_dir = image_encoder.cache_dir

#         self._model_name = self._get_name(args.model)
#         self.model = LinearizedModel(init_model=init_encoder, model=image_encoder)

#     def _get_name(self, model_name):
#         if "__pretrained__" in model_name:
#             model_name, _ = model_name.split("__pretrained__", "")
#         return model_name

#     def forward(self, x):
#         # use the taylorized version of the model.
#         return self.model(x)

#     def __call__(self, x):
#         return self.forward(x)

#     def save(self, filename):
#         """Saves the linearized image encoder.

#         We save the model name in the state dict so that we can load the
#         correct model when loading the linearized image encoder. Directly using
#         torch.save would not work becuse func0 is not serializable.

#         Args:
#             filename (str): The path to save the taylorized image encoder.
#         """
#         if os.path.dirname(filename) != "":
#             os.makedirs(os.path.dirname(filename), exist_ok=True)

#         state_dict = self.state_dict()
#         state_dict["model_name"] = self._model_name

#         torch.save(state_dict, filename)

#     @classmethod
#     def load(cls, filename):
#         """Loads a linearized image encoder.

#         It first loads the state dict with the model name and then creates the
#         correct model and loads the state dict.

#         Args:
#             filename (str): The path to the taylorized image encoder.

#         Returns:
#             LinearizedImageEncoder: The loaded taylorized image encoder.
#         """
#         print(f"Loading image encoder from {filename}")
#         state_dict = torch.load(filename, map_location="cpu")

#         # ImageEncoder expects a DotDict
#         args = DotDict({"model": state_dict["model_name"]})
#         taylorized_encoder = cls(args)

#         # Remove the model name from the state dict so that we can load the
#         # model.
#         state_dict.pop("model_name")
#         taylorized_encoder.load_state_dict(state_dict)
#         return taylorized_encoder