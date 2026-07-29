"""Local inference: causal prompting, padding, slicing, and gated-weight refusal.

Fake tokenizers and models are pushed straight into the module cache and the
torch/transformers runtime accessor is stubbed, so nothing here downloads a
checkpoint, touches the network, or needs requirements-local.txt installed. The
seq2seq anchors are pinned the same way to prove the causal path left them alone.
"""
import contextlib
import datetime

import pytest

from celticbench import hf_local, prompt
from celticbench.registry import get_system, resolve_model, resolve_revision

# Decoded vocabulary: 9 stands for a prompt position, so a missing slice shows up
# as the word PROMPT in the output rather than as a silently plausible sentence.
WORDS = {9: "PROMPT", 50: "Bore", 51: "da.", 52: "nodyn\nolaf"}


class Ids:
    """The 2-D integer tensor surface this module actually uses."""

    def __init__(self, rows):
        self.rows = [list(row) for row in rows]

    @property
    def shape(self):
        return (len(self.rows), len(self.rows[0]) if self.rows else 0)

    def __getitem__(self, key):
        if isinstance(key, tuple):  # generated[:, prompt_width:]
            _, columns = key
            return Ids([row[columns] for row in self.rows])
        return self.rows[key]

    def __iter__(self):
        return iter(self.rows)


class FakeTorch:
    bfloat16 = "bfloat16"

    @staticmethod
    @contextlib.contextmanager
    def inference_mode():
        yield


class FakeBatch(dict):
    """What a tokenizer call returns: a mapping that can be moved to a device."""

    device = None

    def to(self, device):
        self.device = device
        return self


class FakeTokenizer:
    def __init__(self, chat_template=None, pad_token=None, eos_token="</s>", prompt_len=4):
        self.chat_template = chat_template
        self.pad_token = pad_token
        self.eos_token = eos_token
        self.padding_side = "right"
        self.src_lang = None
        self.prompt_len = prompt_len
        self.encoded = []
        self.encode_kwargs = []
        self.decoded = []
        self.template_kwargs = []

    @property
    def pad_token_id(self):
        return None if self.pad_token is None else 7

    @property
    def eos_token_id(self):
        return None if self.eos_token is None else 2

    def __call__(self, batch, **kwargs):
        self.encoded.append(list(batch))
        self.encode_kwargs.append(kwargs)
        rows = [[9] * self.prompt_len for _ in batch]
        return FakeBatch(input_ids=Ids(rows), attention_mask=Ids(rows))

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        assert tokenize is False, "the harness templates to text, then tokenizes the batch"
        assert add_generation_prompt is True, "the model must be cued to answer"
        self.template_kwargs.append(dict(kwargs))
        body = "".join(f"<|user|>{message['content']}<|end|>" for message in messages)
        return f"<bos>{body}<|assistant|>"

    def convert_tokens_to_ids(self, token):
        return 1000 + len(token)

    def batch_decode(self, sequences, skip_special_tokens=False):
        self.decoded.append(sequences)
        return [" ".join(WORDS.get(int(i), "?") for i in row) for row in sequences]


class FakeModel:
    """Returns the prompt columns followed by a fixed continuation, as generate does."""

    def __init__(self, continuation=(50, 51)):
        self.continuation = continuation
        self.generate_kwargs = []

    def generate(self, input_ids=None, attention_mask=None, **kwargs):
        self.generate_kwargs.append(kwargs)
        return Ids([list(row) + list(self.continuation) for row in input_ids])


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    # transformers is None: any cache miss that reaches a loader fails loudly
    # instead of quietly downloading a checkpoint.
    monkeypatch.setattr(hf_local, "_runtime", lambda: (FakeTorch, None, "cpu"))
    hf_local._CACHE.clear()
    yield
    hf_local._CACHE.clear()


def install(system_id, direction, tokenizer, model=None):
    """Seat a fake pair under the exact cache key translate_batch will look up."""
    entry = get_system(system_id)
    model = model if model is not None else FakeModel()
    key = (resolve_model(entry, direction), resolve_revision(entry, direction))
    hf_local._CACHE[key] = (tokenizer, model)
    return entry, model


# ---- seq2seq anchors: unchanged inputs ------------------------------------

def test_opus_cel_prefixes_the_target_language_only_into_celtic():
    tokenizer = FakeTokenizer()
    entry, model = install("opus-mt-cel", "en-xx", tokenizer)
    hf_local.translate_batch(entry, ["Good morning."], "kw", "en-xx")
    assert tokenizer.encoded == [[">>cor<< Good morning."]]
    assert model.generate_kwargs == [dict(entry["decoding"])]
    assert tokenizer.encode_kwargs[0]["max_length"] == entry["context_tokens"]


def test_opus_cel_into_english_sends_the_raw_sentence():
    tokenizer = FakeTokenizer()
    entry, _ = install("opus-mt-cel", "xx-en", tokenizer)
    hf_local.translate_batch(entry, ["Myttin da."], "kw", "xx-en")
    assert tokenizer.encoded == [["Myttin da."]]


def test_madlad_tags_the_target_language_in_both_directions():
    into_celtic = FakeTokenizer()
    entry, _ = install("madlad400-3b", "en-xx", into_celtic)
    hf_local.translate_batch(entry, ["Good morning."], "kw", "en-xx")
    assert into_celtic.encoded == [["<2kw> Good morning."]]

    into_english = FakeTokenizer()
    install("madlad400-3b", "xx-en", into_english)
    hf_local.translate_batch(entry, ["Myttin da."], "kw", "xx-en")
    assert into_english.encoded == [["<2en> Myttin da."]]


def test_nllb_sets_the_source_language_and_forces_the_target_bos():
    tokenizer = FakeTokenizer()
    entry, model = install("nllb-600m", "en-xx", tokenizer)
    hf_local.translate_batch(entry, ["Good morning."], "ga", "en-xx")
    assert tokenizer.src_lang == "eng_Latn"
    assert tokenizer.encoded == [["Good morning."]]
    assert model.generate_kwargs[0]["forced_bos_token_id"] == tokenizer.convert_tokens_to_ids("gle_Latn")

    tokenizer.src_lang = None
    hf_local.translate_batch(entry, ["Dia duit."], "ga", "xx-en")
    assert tokenizer.src_lang == "gle_Latn"
    assert model.generate_kwargs[1]["forced_bos_token_id"] == tokenizer.convert_tokens_to_ids("eng_Latn")


def test_seq2seq_decodes_the_whole_generated_sequence():
    tokenizer = FakeTokenizer(prompt_len=5)
    entry, _ = install("madlad400-3b", "en-xx", tokenizer, FakeModel(continuation=(50, 51)))
    outputs, _ = hf_local.translate_batch(entry, ["Good morning."], "kw", "en-xx")
    assert tokenizer.decoded[0].shape == (1, 7)  # nothing sliced off
    assert outputs[0].startswith("PROMPT")


# ---- causal path ----------------------------------------------------------

def test_causal_sends_the_prompt_unwrapped_when_there_is_no_chat_template():
    tokenizer = FakeTokenizer(chat_template=None, pad_token="<pad>")
    entry, _ = install("qwen3.5-9b", "en-xx", tokenizer)
    hf_local.translate_batch(entry, ["Good morning."], "cy", "en-xx")
    assert tokenizer.encoded == [[prompt.render("cy", "en-xx", "Good morning.")]]
    assert "add_special_tokens" not in tokenizer.encode_kwargs[0]


def test_a_system_with_its_own_template_never_sees_the_shared_prompt():
    tokenizer = FakeTokenizer(chat_template="{{ messages }}", pad_token="<pad>")
    entry, _ = install("salamandrata-7b", "en-xx", tokenizer)
    assert prompt.template_for(entry) != prompt.PROMPT_TEMPLATE, "fixture needs its own template"
    hf_local.translate_batch(entry, ["Good morning."], "ga", "en-xx")
    sent = tokenizer.encoded[0][0]
    own = prompt.render_with(prompt.template_for(entry), "ga", "en-xx", "Good morning.")
    assert sent == f"<bos><|user|>{own}<|end|><|assistant|>"
    # a model the card says was never chat-trained must not be handed our instruction
    assert prompt.render("ga", "en-xx", "Good morning.") not in sent


def test_the_run_date_reaches_only_templates_that_read_it():
    dated = FakeTokenizer(chat_template="{{ date_string }}{{ messages }}", pad_token="<pad>")
    entry, _ = install("salamandrata-7b", "en-xx", dated)
    before = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    hf_local.translate_batch(entry, ["Good morning."], "ga", "en-xx")
    after = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d")
    assert dated.template_kwargs[0]["date_string"] in {before, after}

    undated = FakeTokenizer(chat_template="{{ messages }}", pad_token="<pad>")
    install("salamandrata-7b", "en-xx", undated)
    hf_local.translate_batch(entry, ["Good morning."], "ga", "en-xx")
    assert undated.template_kwargs == [{}]


def test_causal_wraps_the_published_prompt_in_the_models_chat_template():
    tokenizer = FakeTokenizer(chat_template="{{ messages }}", pad_token="<pad>")
    entry, _ = install("qwen3.5-9b", "en-xx", tokenizer)
    hf_local.translate_batch(entry, ["Good morning."], "cy", "en-xx")
    rendered = prompt.render("cy", "en-xx", "Good morning.")
    assert tokenizer.encoded == [[f"<bos><|user|>{rendered}<|end|><|assistant|>"]]
    # the template already emitted BOS and the role markers
    assert tokenizer.encode_kwargs[0]["add_special_tokens"] is False


def test_causal_output_is_only_the_newly_generated_tokens():
    tokenizer = FakeTokenizer(pad_token="<pad>", prompt_len=5)
    entry, _ = install("qwen3.5-9b", "en-xx", tokenizer, FakeModel(continuation=(50, 51, 52)))
    outputs, reported = hf_local.translate_batch(entry, ["Good morning."], "cy", "en-xx")
    assert tokenizer.decoded[0].shape == (1, 3)
    assert outputs == ["Bore da. nodyn olaf"]  # prompt sliced off, newline flattened
    revision = resolve_revision(entry, "en-xx")
    assert reported == f"{resolve_model(entry, 'en-xx')}@{revision[:12]}"


def test_causal_generation_declares_the_pad_token():
    tokenizer = FakeTokenizer(pad_token="<pad>")
    entry, model = install("qwen3.5-9b", "en-xx", tokenizer)
    hf_local.translate_batch(entry, ["Good morning."], "cy", "en-xx")
    assert model.generate_kwargs[0]["pad_token_id"] == tokenizer.pad_token_id
    for key, value in entry["decoding"].items():
        assert model.generate_kwargs[0][key] == value


def test_batches_are_split_and_kept_in_input_order(monkeypatch):
    monkeypatch.setattr(hf_local, "BATCH_SIZE", 2)
    tokenizer = FakeTokenizer(pad_token="<pad>")
    entry, _ = install("qwen3.5-9b", "en-xx", tokenizer)
    outputs, _ = hf_local.translate_batch(entry, ["One.", "Two.", "Three."], "cy", "en-xx")
    assert [len(batch) for batch in tokenizer.encoded] == [2, 1]
    assert tokenizer.encoded[0][0].endswith("One.") and tokenizer.encoded[1][0].endswith("Three.")
    assert len(outputs) == 3


def test_unknown_family_is_refused():
    tokenizer = FakeTokenizer(pad_token="<pad>")
    entry, _ = install("qwen3.5-9b", "en-xx", tokenizer)
    with pytest.raises(SystemExit):
        hf_local.translate_batch({**entry, "family": "rnn"}, ["Good morning."], "cy", "en-xx")


# ---- left padding ---------------------------------------------------------

def test_causal_tokenizer_is_left_padded_and_falls_back_to_eos_for_pad():
    tokenizer = FakeTokenizer(pad_token=None, eos_token="</s>")
    hf_local._prepare_causal_tokenizer(tokenizer, "Qwen/Qwen3.5-9B")
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "</s>"
    assert tokenizer.pad_token_id is not None


def test_an_existing_pad_token_is_kept():
    tokenizer = FakeTokenizer(pad_token="<pad>", eos_token="</s>")
    hf_local._prepare_causal_tokenizer(tokenizer, "Qwen/Qwen3.5-9B")
    assert tokenizer.pad_token == "<pad>"


def test_a_tokenizer_with_neither_pad_nor_eos_is_refused():
    tokenizer = FakeTokenizer(pad_token=None, eos_token=None)
    with pytest.raises(SystemExit) as excinfo:
        hf_local._prepare_causal_tokenizer(tokenizer, "CohereLabs/tiny-aya-water")
    assert "CohereLabs/tiny-aya-water" in str(excinfo.value)


def test_left_padding_stays_on_the_instance_it_was_set_on():
    causal = FakeTokenizer(pad_token=None)
    hf_local._prepare_causal_tokenizer(causal, "Qwen/Qwen3.5-9B")
    anchor = FakeTokenizer()
    entry, _ = install("madlad400-3b", "en-xx", anchor)
    hf_local.translate_batch(entry, ["Good morning."], "kw", "en-xx")
    assert anchor.padding_side == "right"
    assert anchor.pad_token is None


# ---- gated weights --------------------------------------------------------

def test_gated_weights_are_refused_without_a_token(monkeypatch):
    entry = get_system("tiny-aya-water")
    monkeypatch.delenv(entry["token_env"], raising=False)
    with pytest.raises(SystemExit) as excinfo:
        hf_local.translate_batch(entry, ["Good morning."], "cy", "en-xx")
    message = str(excinfo.value)
    assert entry["model"] in message
    assert entry["license"] in message
    assert entry["token_env"] in message


def test_a_blank_token_is_not_treated_as_a_token(monkeypatch):
    entry = get_system("translategemma-4b")
    monkeypatch.setenv(entry["token_env"], "   ")
    with pytest.raises(SystemExit):
        hf_local._gated_token(entry, entry["model"])


def test_the_gated_token_comes_from_the_declared_variable(monkeypatch):
    entry = get_system("translategemma-4b")
    monkeypatch.setenv(entry["token_env"], "hf_secret")
    assert hf_local._gated_token(entry, entry["model"]) == "hf_secret"


def test_ungated_weights_need_no_token():
    entry = get_system("salamandrata-7b")
    assert hf_local._gated_token(entry, entry["model"]) is None


# ---- loader chain ---------------------------------------------------------

class FakeConfig:
    def __init__(self, architectures=(), sub_configs=None):
        self.architectures = list(architectures)
        self.sub_configs = dict(sub_configs or {})


class FakeAutoConfig:
    def __init__(self, config):
        self.config = config

    def from_pretrained(self, model_id, **kwargs):
        return self.config


class FakeLoader:
    def __init__(self, built=None, error=None):
        self.built = built
        self.error = error
        self.calls = []

    def from_pretrained(self, model_id, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.built


class FakeTransformers:
    __version__ = "5.14.1-fake"

    def __init__(self, config, causal, image_text):
        self.AutoConfig = FakeAutoConfig(config)
        self.AutoModelForCausalLM = causal
        self.AutoModelForImageTextToText = image_text


def built_as(architecture):
    return type(architecture, (), {})()


def test_a_text_only_checkpoint_loads_through_the_causal_class():
    causal = FakeLoader(built=built_as("LlamaForCausalLM"))
    image_text = FakeLoader(built=built_as("LlamaForCausalLM"))
    fake = FakeTransformers(FakeConfig(["LlamaForCausalLM"]), causal, image_text)
    model = hf_local._causal_model(fake, "BSC-LT/salamandraTA-7b-instruct", "67e0f70c3a05",
                                   {"revision": "67e0f70c3a05"}, "bfloat16")
    assert type(model).__name__ == "LlamaForCausalLM"
    assert image_text.calls == []
    assert causal.calls == [{"revision": "67e0f70c3a05", "dtype": "bfloat16"}]


def test_a_composite_checkpoint_skips_the_text_only_causal_class():
    # Qwen3.5-9B nests its decoder under text_config; AutoModelForCausalLM would
    # resolve to Qwen3_5ForCausalLM and quietly load that tower alone.
    config = FakeConfig(["Qwen3_5ForConditionalGeneration"],
                        {"text_config": object, "vision_config": object})
    causal = FakeLoader(built=built_as("Qwen3_5ForCausalLM"))
    image_text = FakeLoader(built=built_as("Qwen3_5ForConditionalGeneration"))
    fake = FakeTransformers(config, causal, image_text)
    model = hf_local._causal_model(fake, "Qwen/Qwen3.5-9B", "c202236235762",
                                   {"revision": "c202236235762"}, "bfloat16")
    assert type(model).__name__ == "Qwen3_5ForConditionalGeneration"
    assert causal.calls == []
    assert image_text.calls == [{"revision": "c202236235762", "dtype": "bfloat16"}]


def test_the_loader_falls_back_to_image_text_to_text():
    config = FakeConfig(["Gemma3ForConditionalGeneration"])
    causal = FakeLoader(error=ValueError("Unrecognized configuration class"))
    image_text = FakeLoader(built=built_as("Gemma3ForConditionalGeneration"))
    fake = FakeTransformers(config, causal, image_text)
    model = hf_local._causal_model(fake, "google/translategemma-4b-it", "10042cb0e6e7", {}, "bfloat16")
    assert type(model).__name__ == "Gemma3ForConditionalGeneration"
    assert len(causal.calls) == 1 and len(image_text.calls) == 1


def test_a_class_contradicting_the_declared_architecture_is_rejected():
    config = FakeConfig(["Qwen3_5ForConditionalGeneration"])
    causal = FakeLoader(built=built_as("Qwen3_5ForCausalLM"))
    image_text = FakeLoader(built=built_as("Qwen3_5ForConditionalGeneration"))
    fake = FakeTransformers(config, causal, image_text)
    model = hf_local._causal_model(fake, "Qwen/Qwen3.5-9B", "c202236235762", {}, "bfloat16")
    assert type(model).__name__ == "Qwen3_5ForConditionalGeneration"


def test_no_working_loader_names_the_model_and_both_failures():
    config = FakeConfig(["Qwen3_5ForConditionalGeneration"], {"text_config": object})
    fake = FakeTransformers(config,
                            FakeLoader(error=ValueError("no text weights")),
                            FakeLoader(error=OSError("no processor")))
    revision = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
    with pytest.raises(SystemExit) as excinfo:
        hf_local._causal_model(fake, "Qwen/Qwen3.5-9B", revision, {}, "bfloat16")
    message = str(excinfo.value)
    assert f"Qwen/Qwen3.5-9B@{revision[:12]}" in message
    assert "OSError" in message and "ValueError" in message
