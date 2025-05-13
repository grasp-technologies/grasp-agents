import logging
from collections.abc import Iterator, Sequence
from copy import deepcopy

from .typing.io import LLMPrompt
from .typing.message import Conversation, Message, SystemMessage

logger = logging.getLogger(__name__)


class MessageHistory:
    def __init__(self, sys_prompt: LLMPrompt | None = None) -> None:
        self._sys_prompt = sys_prompt
        self._batched_conversations: list[Conversation]
        self.reset()

    @property
    def sys_prompt(self) -> LLMPrompt | None:
        return self._sys_prompt

    def add_message_batch(self, message_batch: Sequence[Message]) -> None:
        """
        Adds a batch of messages to the current batched conversations.
        This method verifies that the size of the input message batch matches
        the expected batch size (self.batch_size).
        If there is a mismatch, the method adjusts by duplicating either
        the message or the conversation as necessary:

        - If the message batch contains exactly one message and
            self.batch_size > 1, the single message is duplicated to match
            the batch size.
        - If the message batch contains multiple messages but
            self.batch_size == 1, the entire conversation is duplicated to
            accommodate each message in the batch.
        - If the message batch size does not match self.batch_size and none of
            the above adjustments apply, a ValueError is raised.

        Afterwards, each message in the batch is appended to its corresponding
        conversation in the batched conversations.

        Args:
            message_batch: A sequence of Message objects
                representing the batch of messages to be added. Must align with
                or be adjusted to match the current batch size.

        Raises:
            ValueError: If the message batch size does not match the current
                batch size and cannot be automatically adjusted.

        """
        message_batch_size = len(message_batch)

        if message_batch_size == 1 and self.batch_size > 1:
            logger.info(
                "Message batch size is 1, current batch size is "
                f"{self.batch_size}: duplicating the message to match the "
                "current batch size"
            )
            message_batch = self._duplicate_message_to_current_batch_size(message_batch)
            message_batch_size = self.batch_size
        elif message_batch_size > 1 and self.batch_size == 1:
            logger.info(
                f"Message batch size is {len(message_batch)}, current batch "
                "size is 1: duplicating the conversation to match the message "
                "batch size"
            )
            self._duplicate_conversation_to_message_batch_size(message_batch_size)
        elif message_batch_size != self.batch_size:
            raise ValueError(
                f"Message batch size {message_batch_size} does not match "
                f"current batch size {self.batch_size}"
            )

        for batch_id in range(message_batch_size):
            self._batched_conversations[batch_id].append(message_batch[batch_id])

    def add_message_batches(self, message_batches: Sequence[Sequence[Message]]) -> None:
        for message_batch in message_batches:
            self.add_message_batch(message_batch)

    def add_message(self, message: Message) -> None:
        for conversation in self._batched_conversations:
            conversation.append(message)

    def add_messages(self, messages: Sequence[Message]) -> None:
        for message in messages:
            self.add_message(message)

    def __len__(self) -> int:
        return len(self._batched_conversations[0])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(len={len(self)}; bs={self.batch_size})"

    def __getitem__(self, idx: int) -> tuple[Message, ...]:
        return tuple(conversation[idx] for conversation in self._batched_conversations)

    def __iter__(self) -> Iterator[tuple[Message, ...]]:
        for idx in range(len(self)):
            yield tuple(
                conversation[idx] for conversation in self._batched_conversations
            )

    def _duplicate_message_to_current_batch_size(
        self, message_batch: Sequence[Message]
    ) -> Sequence[Message]:
        assert len(message_batch) == 1, (
            "Message batch size must be 1 to duplicate to current batch size"
        )

        return [deepcopy(message_batch[0]) for _ in range(self.batch_size)]

    def _duplicate_conversation_to_message_batch_size(
        self, target_batch_size: int
    ) -> None:
        assert self.batch_size == 1, "Batch size must be 1 to duplicate conversation"
        self._batched_conversations = [
            deepcopy(self._batched_conversations[0]) for _ in range(target_batch_size)
        ]

    @property
    def batched_conversations(self) -> list[Conversation]:
        return self._batched_conversations

    @property
    def batch_size(self) -> int:
        return len(self._batched_conversations)

    def reset(
        self, sys_prompt: LLMPrompt | None = None, *, batch_size: int = 1
    ) -> None:
        if sys_prompt is not None:
            self._sys_prompt = sys_prompt

        conv: Conversation
        if self._sys_prompt is not None:
            conv = [SystemMessage(content=self._sys_prompt)]
        else:
            conv = []

        self._batched_conversations = [deepcopy(conv) for _ in range(batch_size)]

    def erase(self) -> None:
        self._batched_conversations = [[]]
