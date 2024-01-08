from cjw.knowledgeqa.bots.GptBot import GptBot


def bot(model: str, **kwargs):
    if model.lower() in GptBot.models:
        return GptBot.of(model=model, **kwargs)