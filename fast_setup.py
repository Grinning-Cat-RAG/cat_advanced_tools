from cat import hook, RecallSettings


@hook
async def agent_prompt_prefix(prefix: str, cat) -> str:
    settings = await cat.mad_hatter.get_plugin().load_settings()
    prefix = settings["prompt_prefix"]

    # Escape curly braces to prevent LangChain PromptTemplate from
    # interpreting them as format placeholders (e.g. JSON {…} in the
    # custom prompt would otherwise cause KeyError at LLM invocation).
    prefix = prefix.replace("{", "{{").replace("}", "}}")

    return prefix


@hook
async def before_cat_recalls_memories(config: RecallSettings, cat) -> RecallSettings:
    settings = await cat.mad_hatter.get_plugin().load_settings()
    config.k = settings["k"]
    config.threshold = settings["threshold"]
    config.latest_n_history = settings["latest_n_history"]

    return config


@hook
async def agent_prompt_suffix(suffix: str, cat) -> str:
    settings = await cat.mad_hatter.get_plugin().load_settings()
    suffix = f"""
# Context
{{context}}
"""

    language = settings["language"].value.lower()
    return f"""
{suffix}
ALWAYS answer in {"the user's language" if language == "human" else language}
"""
