import pytest

from providers.factory import build_llm_chain, build_stt_chain, build_tts_chain


def test_build_tts_chain_cartesia(has_cartesia):
    if not has_cartesia:
        pytest.skip("CARTESIA_API_KEY not set")
    chain = build_tts_chain()
    assert len(chain.providers) >= 1


def test_build_stt_chain_cartesia(has_cartesia):
    if not has_cartesia:
        pytest.skip("CARTESIA_API_KEY not set")
    chain = build_stt_chain()
    assert len(chain.providers) == 1


@pytest.mark.asyncio
async def test_cartesia_tts_stream(has_cartesia):
    if not has_cartesia:
        pytest.skip("CARTESIA_API_KEY not set")
    chain = build_tts_chain()
    stream, name = chain.get_stream("synthesize_stream", "Hello, this is a test.")
    chunks = []
    async for c in stream:
        chunks.append(c)
        if len(chunks) > 2:
            break
    assert len(chunks) >= 1, f"No audio from {name}"


@pytest.mark.asyncio
async def test_gemini_stream(has_gemini):
    if not has_gemini:
        pytest.skip("GEMINI_API_KEY not set")
    chain = build_llm_chain()
    last_err = None
    for provider in chain.providers:
        try:
            stream = provider.stream("Say hello in five words.")
            tokens = []
            async for t in stream:
                tokens.append(t)
                if len("".join(tokens)) > 5:
                    break
            assert len(tokens) >= 1
            return
        except Exception as e:
            last_err = e
            err = str(e)
            if "429" in err or "Quota" in err or "404" in err or "not found" in err.lower():
                continue
            raise
    if last_err:
        pytest.skip(f"All Gemini keys unavailable (quota/model): {last_err}")