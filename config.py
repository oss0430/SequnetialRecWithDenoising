from transformers import BartConfig

class BARTforSeqRecConfig(BartConfig):
    def __init__(
        self,
        **common_kwargs
    ):
        super().__init__(
            **common_kwargs
        )
