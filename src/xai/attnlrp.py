from transformers.models.llama import modeling_llama
from transformers.models.qwen2 import modeling_qwen2
from transformers import AutoTokenizer, AutoModelForCausalLM
from lxt.efficient import monkey_patch
import torch


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

    # def _attribute_text(
    #     self, prompt: str, forced_output: str = None, heatmap_file: str = "heatmap.pdf"
    # ):

    #     # Get input embeddings
    #     input_ids = self.tokenizer(
    #         prompt,
    #         return_tensors="pt",
    #         add_special_tokens=False,
    #     ).input_ids.to(self.model.device)

    #     # TODO: techincally, we should add EOS here
    #     output_token_ids = self.tokenizer(
    #         forced_output, add_special_tokens=False, return_tensors="pt"
    #     ).input_ids
    #     # add EOS token
    #     output_token_ids = torch.cat(
    #         [output_token_ids, torch.tensor([[self.tokenizer.eos_token_id]])], dim=-1
    #     )
    #     output_token_ids = output_token_ids.to(self.model.device)

    #     # import pdb

    #     # pdb.set_trace()
    #     fwd_ids = torch.cat([input_ids, output_token_ids], dim=-1)

    #     logits_to_keep = output_token_ids.shape[1]
    #     input_embeds = self.model.get_input_embeddings()(fwd_ids)

    #     # Inference
    #     output_logits = self.model(
    #         inputs_embeds=input_embeds.requires_grad_(),
    #         use_cache=True,
    #         logits_to_keep=logits_to_keep + 1,
    #     ).logits
    #     output_logits = output_logits[:, -(logits_to_keep + 1) : -1, :]

    #     # in theory, this argmax should return the same values of output_token_ids
    #     # however, here we are using HF while we generated the translations with vllm
    #     # so we force-pick the token ids coming from the forced_output
    #     # pred_ids = output_logits.argmax(dim=-1) # TODO: we are force-picking pred_ids instead
    #     indices = output_token_ids.unsqueeze(-1)  # shape: (1, seq_len, 1)
    #     pred_ids = torch.gather(output_logits, dim=-1, index=indices)
    #     grad = torch.ones_like(pred_ids)
    #     pred_ids.backward(gradient=grad)

    #     # pred_ids = output_logits.argmax(-1)
    #     # cumulative_relevance = list()
    #     # for i, token_id in enumerate(output_token_ids[0]):
    #     #     logit = output_logits[0, i, token_id]
    #     #     # Backward pass (the relevance is initialized with the value of max_logits)
    #     #     logit.backward(retain_graph=True)

    #     # Obtain relevance. (Works at any layer in the model!)
    #     relevance = (
    #         (input_embeds.grad * input_embeds).float().sum(-1).detach().cpu()
    #     )  # Cast to float32 for higher precision

    #     # # Normalize relevance between [-1, 1]
    #     # # relevance = relevance / relevance.abs().max()
    #     # cumulative_relevance.append(relevance.detach().cpu())

    #     # # clear input_embeds.grad
    #     # input_embeds.grad.zero_()

    #     # relevance = torch.stack(cumulative_relevance, dim=0)
    #     # relevance = relevance.mean(0)

    #     relevance = relevance[0, : len(input_ids[0])]

    #     # Normalize relevance between -1 and 1
    #     relevance = relevance / relevance.abs().max()

    #     # Remove special characters that are not compatible wiht LaTeX
    #     tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
    #     # import pdb
    #     # pdb.set_trace()
    #     # tokens = clean_tokens(tokens)

    #     # Save heatmap as PDF
    #     # pdf_heatmap(
    #     #     tokens,
    #     #     relevance,
    #     #     path=heatmap_file,
    #     #     backend="xelatex",
    #     #     append_string=rf"\\ \\Translation: \\ \\{forced_output}",
    #     # )
