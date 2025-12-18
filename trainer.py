"""
Minimal subclasses of GKDConfig and GKDTrainer to add min_new_tokens support.

The upstream trl GKDConfig doesn't expose min_new_tokens, so we subclass both
the config (to accept the arg) and the trainer (to pass it to generation_kwargs).
"""

from dataclasses import dataclass

from trl.experimental.gkd import GKDConfig, GKDTrainer


@dataclass
class MinTokensGKDConfig(GKDConfig):
    min_new_tokens: int = 1


class MinTokensGKDTrainer(GKDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_config.min_new_tokens = self.args.min_new_tokens
        self.generation_kwargs["min_new_tokens"] = self.args.min_new_tokens
