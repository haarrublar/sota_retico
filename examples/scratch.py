from retico_huggingfacelm import HuggingfaceLMClient

_llm_client = HuggingfaceLMClient.quick_from_checkpoint("HuggingFaceTB/SmolLM2-1.7B-Instruct", device="cuda:0", temperature=0.7)
_llm_client.set_system_role("Determine the requested attribute from the given text. "
    "Output ONLY the requested value, or <UNKNOWN> if it is not mentioned. "
    "Example 1:\nRequest: job\nText: My name is Alice.\nOutput: <UNKNOWN>\n"
    "Example 2:\nRequest: job\nText: I work as a software engineer.\nOutput: software engineer")
response = _llm_client.generate_response("Extract the person's job. Text: Hi there, my name is Jim Young")
print(response)