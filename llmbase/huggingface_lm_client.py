import threading

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    TextStreamer,
)


class HuggingfaceLMClient:
    @classmethod
    def quick_from_checkpoint(
        cls,
        checkpoint,
        device="cpu",
        system_role="You are a friendly chatbot who responds to questions",  # just a default.
        max_history_turns=10,  # how many exchange turns (q then a) to keep and feed to the model.
        max_new_tokens=75,  # how many tokens (~words) it can produce at max
        temperature=0.2,  # between 0 and 1, higher means more randomness
    ):
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint, trust_remote_code=True
        ).to(device)  # type: ignore
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        return cls(
            device,
            tokenizer,
            model,
            streamer,
            system_role,
            max_history_turns,
            max_new_tokens,
            temperature,
        )

    def __init__(
        self,
        device,
        tokenizer,
        model,
        streamer,
        system_role,
        max_history_turns,
        max_new_tokens,
        temperature,
    ):
        self._device = device
        self._tokenizer = tokenizer
        self._model = model
        self._streamer = streamer
        self._max_history_turns = max_history_turns
        self._message_history = []
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self.turns = 0
        self.set_system_role(system_role)

    ########## API interface to the LM
    # - set the system role, defines the instructions to the LM (the system role)
    # - set max history turns, defines how many exchanges to keep in the prompt to the LM
    # - clear history, as described, keeps the role
    # - generate response, takes a next query, compiles the role and history, and returns the result.
    # - stream response (added), same as generate response but gives the text while it is written

    # set the system role to be sent into the LLM from this point onward. Does not change history.
    def set_system_role(self, role):
        if len(self._message_history) == 0:
            self._message_history.append({"role": "system", "content": role})
        else:
            self._message_history[0] = {"role": "system", "content": role}

    # set maximum number of exchange turns (prompt then response) to keep in history and feed to the model
    # If we have history allready and the newe limit is > existing history, truncates
    def set_max_history_turns(self, turns):
        self._max_history_turns = turns
        if turns < self.turns and turns > 0:  # truncate, keep the newest n turns
            # self._message_history = [self._message_history[0]] + self._messages_history[-(turns * 2) :]
            # typo, _messages_history does not exist
            self._message_history = [self._message_history[0]] + self._message_history[
                -(turns * 2) :
            ]
            self.turns = turns  # added: keep the counter in sync after cutting

    def clear_history(self):
        system_role = self._message_history[0]["content"]
        self._message_history = []
        self.set_system_role(system_role)

    def _add_response_and_truncate(self, response):
        self._message_history.append({"role": "assistant", "content": response})
        self.turns += 1

        if self.turns > self._max_history_turns:
            self.turns -= 1
            self._message_history = [self._message_history[0]] + self._message_history[
                -(self.turns * 2) :
            ]

    # aux function, same tokenizing as generate_response, used by both
    def _prepare_input(self, query, keep_history=True):
        if not keep_history:
            self.clear_history()
        self._message_history.append({"role": "user", "content": query})

        tokenized_chat = self._tokenizer.apply_chat_template(
            self._message_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )

        if isinstance(
            tokenized_chat, torch.Tensor
        ):  # sometimes we get a tensor, other times a dictionary, based on LM version
            return tokenized_chat.to(self._device)
        return tokenized_chat["input_ids"].to(self._device)

    def generate_response(self, query, keep_history=True):

        # if not keep_history:
        #     self.clear_history()
        # self._message_history.append({"role": "user", "content": query})
        #
        # tokenized_chat = self._tokenizer.apply_chat_template(
        #     self._message_history,
        #     tokenize=True,
        #     add_generation_prompt=True,
        #     return_tensors="pt",
        # )
        #
        # if isinstance(tokenized_chat, torch.Tensor):
        #     input_ids = tokenized_chat.to(self._device)
        # else:
        #     input_ids = tokenized_chat["input_ids"].to(self._device)
        input_ids = self._prepare_input(query, keep_history)  # added

        input_length = input_ids.shape[1]

        with torch.no_grad():
            output_tokens = self._model.generate(
                input_ids,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature,
                top_p=0.9,
                do_sample=True,
                streamer=self._streamer,
            )

        response = self._tokenizer.decode(
            output_tokens[0][input_length:], skip_special_tokens=True
        )
        self._add_response_and_truncate(response)
        return response

    # same as generate_response, but gives the reply piece by piece
    # while the model is writing it, so the tts can start before the end
    def stream_response(self, query, keep_history=True):
        input_ids = self._prepare_input(query, keep_history)

        streamer = TextIteratorStreamer(
            self._tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        # generate runs in a thread, so I can read the text while it is written
        threading.Thread(
            target=self._model.generate,
            kwargs={
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "max_new_tokens": self._max_new_tokens,
                "temperature": self._temperature,
                "top_p": 0.9,
                "do_sample": True,
                "streamer": streamer,
            },
            daemon=True,
        ).start()

        response = ""
        for text in streamer:
            print(text, end="", flush=True)  # to see the reply in the terminal
            response += text
            yield text
        print()
        self._add_response_and_truncate(response.strip())  # keep the history
