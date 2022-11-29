from transformers import BartConfig

class BARTforSeqRecConfig(BartConfig):
    def __init__(
        self,
        mask_token_id = 50264,
        **common_kwargs
    ):
        super().__init__(
            **common_kwargs
        )
        self.mask_token_id = mask_token_id