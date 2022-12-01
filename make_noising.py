import numpy as np
import torch

def make_train_examples(source_tokens: torch.Tensor, target_tokens: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """ Make traning examples from source and targe tokens. """
    return {
        "input_ids": source_tokens,
        "attention_mask": torch.ones(source_tokens, dtype=torch.int32),
        "decoder_input_ids": target_tokens[:-1],
    }, target_tokens[1:]


def text_infilling(mask_token_id: int, masking_rate: float = 0.3):
    mask_token = torch.tensor([mask_token_id], dtype=torch.int32)

    input_signature = [
        {
            "input_ids": torch.tensor([None], dtype=torch.int32),
            "attention_mask": torch.tensor([None], dtype=torch.int32),
            "decoder_input_ids": torch.tensor([None], dtype=torch.int32)
        },
        torch.tensor([None], dtype=torch.int32)
    ]

    def _text_infilling(inputs: dict[str, torch.tensor], target: torch.tensor) -> tuple[dict[str, torch.tensor], torch.tensor]:
        """ Add text infilling noise to example """
        source_tokens = inputs["input_ids"]
        token_length = torch.Size(source_tokens)[0]
        masking_length = torch.int32(torch.float32(token_length) * masking_rate)
        masked_length = 0

        while masked_length < masking_length:
            span_length = torch.minimum(torch.from_numpy(np.random.poisson((), lam=3), dtype=torch.int32), token_length - 1)
            start_index = torch.from_numpy(np.random.uniform((), 0, token_length - span_length), dtype=torch.int32)

            source_tokens = torch.concat(
                [
                    source_tokens[:start_index],
                    mask_token,
                    source_tokens[start_index + span_length :]
                ],
                axis = 0
            )
            token_length -= span_length - 1
            masked_length += span_length

        return {
            "input_ids": source_tokens,
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": inputs["decoder_input_ids"]
        }, target
    
    return _text_infilling


def sentence_permutation(segment_token_id: int):
    input_signature = [
        {
            "input_ids": torch.tensor([None], dtype=torch.int32),
            "attention_mask": torch.tensor([None], dtype=torch.int32),
            "decoder_input_ids": torch.tensor([None], dtype=torch.int32)
        },
        torch.tensor([None], dtype=torch.int32)
    ]

    def _sentence_permutation(
        inputs: dict[str, torch.tensor], target: torch.tensor
    ) -> tuple[dict[str, torch.tensor], torch.tensor]:
        """ Permute by segment token ID """
        source_tokens = inputs["input_ids"]
        num_source_tokens = torch.int64(source_tokens)

        is_segment = source_tokens == segment_token_id
        segment_end_indices = torch.concat([torch.squeeze(torch.where(is_segment), axis=1), num_source_tokens], axis=0)
        segment_start_indices = torch.concat([[0], segment_end_indices[:-1] + 1], axis=0)
        segment_indices = torch.stack([segment_start_indices, segment_end_indices], axis=1)
        shuffled_segment_indices = torch.randperm(segment_indices)

        first_segment = shuffled_segment_indices[0]
        shuffled_segment_indices = shuffled_segment_indices[1:]
        permutated_source_tokens = source_tokens[first_segment[0] : first_segment[1]]

        num_segments = torch.Size(shuffled_segment_indices)[0]
        for i in torch.range(num_segments):
            indices = shuffled_segment_indices[i]
            segment = source_tokens[indices[0] : indices[1]]
            permutated_source_tokens = torch.concat([permutated_source_tokens, [segment_token_id], segment], axis=0)

        return {
            "input_ids": permutated_source_tokens,
            "attention_mask": inputs["attention_mask"],
            "decoder_input_ids": inputs["decoder_input_ids"]
        }, target
    
    return _sentence_permutation


def filter_example(max_sequence_length: int) -> callable[[torch.tensor, torch.tensor], torch.tensor]:
    def _filter(source_tokens: torch.tensor, target_tokens: torch.tensor) -> torch.tensor:
        return torch.logical_and(
            torch.Tensor.size(source_tokens) < max_sequence_length,
            torch.Tensor.size(target_tokens) < max_sequence_length
        )
    return _filter


def slice_example(max_sequence_length: int) -> callable[[torch.tensor, torch.tensor], tuple[torch.tensor, torch.tensor]]:
    def _slice(source_tokens: torch.tensor, target_tokens: torch.tensor) -> tuple[torch.tensor, torch.tensor]:
        return (
            source_tokens[:max_sequence_length],
            target_tokens[:max_sequence_length]
        )
    return _slice