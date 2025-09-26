import torch
from lxt.efficient import monkey_patch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2


class AttLRPHelper:
    """Class inspired by: https://lxt.readthedocs.io/en/latest/quickstart.html"""

    def __init__(self, model_name_or_path: str):
        # Modify the LLaMA module to compute LRP in the backward pass
        # monkey_patch(modeling_llama, verbose=True)

        if "qwen" in model_name_or_path.lower():
            monkey_patch(modeling_qwen2, verbose=True)
        elif "llama" in model_name_or_path.lower():
            monkey_patch(modeling_llama, verbose=True)
        else:
            raise ValueError("This model can't be patched yet.")

        self.model_name_or_path = model_name_or_path
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.train()
        # self.model.gradient_checkpointing_enable()

    def tokenize(self, x: str):
        return self.tokenizer.encode(x)

    def get_attributions(
        self,
        input_ids: list[int],
        output_ids: list[int],
        attributed_span: tuple[int, int],
        attributing_span: tuple[int, int],
    ):
        """Compute attributions using LRP.

        Args:
            input_ids (list[int]): Input token IDs.
            output_ids (list[int]): Output token IDs.
            attributed_span (tuple[int, int]): Span of tokens to attribute.
            attributing_span (tuple[int, int]): Span of tokens to attribute from.
        """
        assert len(attributing_span) == len(attributed_span) == 2

        input_ids = torch.tensor([input_ids], device=self.model.device)
        output_ids = torch.tensor(
            [output_ids + [self.tokenizer.eos_token_id]],
            device=self.model.device,
        )
        fwd_ids = torch.cat([input_ids, output_ids], dim=-1)

        logits_to_keep = output_ids.shape[1]
        input_embeds = self.model.get_input_embeddings()(fwd_ids)

        # inference
        output_logits = self.model(
            inputs_embeds=input_embeds.requires_grad_(),
            use_cache=True,
            logits_to_keep=logits_to_keep + 1,
        ).logits
        output_logits = output_logits[:, -(logits_to_keep + 1) : -1, :]

        indices = output_ids.unsqueeze(-1)  # shape: (1, seq_len, 1)
        indices = indices[:, attributing_span[0] : attributing_span[1], :]

        pred_logits = torch.gather(output_logits, dim=-1, index=indices)
        pred_logits = pred_logits.sum()
        pred_logits.backward()

        # Obtain relevance. (Works at any layer in the model!)
        relevance = (
            (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()
        )  # Cast to float32 for higher precision

        relevance = relevance[0, attributed_span[0] : attributed_span[1]]
        return relevance
