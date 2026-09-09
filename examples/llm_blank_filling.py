from retico_huggingfacelm import HuggingfaceLMClient

_llm_client = HuggingfaceLMClient.quick_from_checkpoint("HuggingFaceTB/SmolLM2-1.7B-Instruct", device="cuda:0", temperature=0.7)
_llm_client.set_system_role("Determine the requested attribute from the given text. "
    "Output ONLY the requested value, or <UNKNOWN> if it is not mentioned. ")

few_shot_prompt = (
    "Example 1:\n"
    "Request: job\n"
    "Text: My name is Alice.\n"
    "Output: <UNKNOWN>\n\n"
    "Example 2:\n"
    "Request: job\n"
    "Text: I work as a software engineer.\n"
    "Output: software engineer\n\n"
    "Request: {request}\n"
    "Text: {text}\n"
    "Output:"
)

text = "Hi there, I'm Jim and I am a lawyer"
print( _llm_client.generate_response(few_shot_prompt.format(request="job", text=text), keep_history=False) )
print( _llm_client.generate_response(few_shot_prompt.format(request="name", text=text), keep_history=False) )
print( _llm_client.generate_response(few_shot_prompt.format(request="name", text=text), keep_history=False) )