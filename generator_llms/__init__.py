# Lazy imports to avoid import errors when credentials are not available
def get_claude_response(prompt):
    from .claude_api import get_claude_response as _get_claude_response
    return _get_claude_response(prompt)

def gpt_chat_35_msg(*args, **kwargs):
    from .gpt_azure import gpt_chat_35_msg as _gpt_chat_35_msg
    return _gpt_chat_35_msg(*args, **kwargs)

def gpt_chat_4omini(*args, **kwargs):
    from .gpt_azure import gpt_chat_4omini as _gpt_chat_4omini
    return _gpt_chat_4omini(*args, **kwargs)

def gpt_chat_4o(*args, **kwargs):
    from .gpt_azure import gpt_chat_4o as _gpt_chat_4o
    return _gpt_chat_4o(*args, **kwargs)
