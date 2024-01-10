from cjw.knowledgeqa.bots.Bot import Bot
from cjw.knowledgeqa.bots.GptBot import GptBot


def bot(model: str, **kwargs) -> Bot:
    """Create a Bot with a model name"""
    if model.lower() in GptBot.models:
        return GptBot.of(model=model, **kwargs)
    else:
        raise NotImplementedError(f"Model {model} not supported")
