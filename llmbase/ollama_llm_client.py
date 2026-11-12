import ollama


class OllamaLMClient:
    @classmethod
    def quick_from_checkpoint(
        cls,
        checkpoint,  # an Ollama model name, e.g. "qwen2.5:3b"
        device=None,  # unused: Ollama decides; kept so main.py doesn't change
        system_role="You are a friendly chatbot who responds to questions",
        max_history_turns=10,
        max_new_tokens=75,
        temperature=0.2,
    ):
        return cls(
            checkpoint, system_role, max_history_turns, max_new_tokens, temperature
        )

    def __init__(
        self, model, system_role, max_history_turns, max_new_tokens, temperature
    ):
        self._model = model
        self._max_history_turns = max_history_turns
        self._max_new_tokens = max_new_tokens
        self._temperature = temperature
        self._message_history = []
        self.turns = 0
        self.set_system_role(system_role)

    def set_system_role(self, role):
        if len(self._message_history) == 0:
            self._message_history.append({"role": "system", "content": role})
        else:
            self._message_history[0] = {"role": "system", "content": role}

    def set_max_history_turns(self, turns):
        self._max_history_turns = turns
        if 0 < turns < self.turns:
            self._message_history = [self._message_history[0]] + self._message_history[
                -(turns * 2) :
            ]
            self.turns = turns

    def clear_history(self):
        system_role = self._message_history[0]["content"]
        self._message_history = []
        self.turns = 0
        self.set_system_role(system_role)

    def _add_response_and_truncate(self, response):
        self._message_history.append({"role": "assistant", "content": response})
        self.turns += 1
        if self.turns > self._max_history_turns:
            self.turns -= 1
            self._message_history = [self._message_history[0]] + self._message_history[
                -(self.turns * 2) :
            ]

    def _options(self):
        return {
            "num_predict": self._max_new_tokens,
            "temperature": self._temperature,
            "top_p": 0.9,
        }

    def generate_response(self, query, keep_history=True):
        if not keep_history:
            self.clear_history()
        self._message_history.append({"role": "user", "content": query})
        reply = ollama.chat(
            model=self._model,
            messages=self._message_history,
            options=self._options(),
        )
        response = reply["message"]["content"].strip()
        self._add_response_and_truncate(response)
        return response

    def stream_response(self, query, keep_history=True):
        if not keep_history:
            self.clear_history()
        self._message_history.append({"role": "user", "content": query})

        response = ""
        for chunk in ollama.chat(
            model=self._model,
            messages=self._message_history,
            options=self._options(),
            stream=True,
        ):
            text = chunk["message"]["content"]
            print(text, end="", flush=True)
            response += text
            yield text
        print()
        self._add_response_and_truncate(response.strip())
