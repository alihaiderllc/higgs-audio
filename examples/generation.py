# -*- coding: utf-8 -*-
"""High-throughput multithreaded voiceover generator using HiggsAudio (single model load, true batched GPU inference)."""

import click
import soundfile as sf
import langid
import jieba
import os
import re
import copy
import json
import torchaudio  # noqa: F401  (kept for environment consistency if needed by deps)
import tqdm
import yaml
import math
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

from loguru import logger
from typing import List, Optional, Tuple
from dataclasses import asdict

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse  # noqa: F401
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent
from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern

# -------------------- Global Constants --------------------
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""

# -------------------- Text Utilities --------------------
def normalize_chinese_punctuation(text: str) -> str:
    mapping = {
        "：": ":", "；": ";", "？": "?", "！": "!", "（": "(", "）": ")",
        "【": "[", "】": "]", "《": "<", "》": ">", "“": '"', "”": '"', "‘": "'", "’": "'",
        "、": ",", "—": "-", "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    mapping["，"] = ", "
    mapping["。"] = ". "
    for zh, en in mapping.items():
        text = text.replace(zh, en)
    return text


def prepare_chunk_text(text: str, chunk_method: Optional[str] = None,
                       chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1) -> List[str]:
    if chunk_method is None:
        return [text]
    elif chunk_method == "speaker":
        lines = text.split("\n")
        speaker_chunks, speaker_utterance = [], ""
        for line in lines:
            line = line.strip()
            if line.startswith("[SPEAKER") or line.startswith("<|speaker_id_start|>"):
                if speaker_utterance:
                    speaker_chunks.append(speaker_utterance.strip())
                speaker_utterance = line
            else:
                speaker_utterance = (speaker_utterance + "\n" + line) if speaker_utterance else line
        if speaker_utterance:
            speaker_chunks.append(speaker_utterance.strip())
        if chunk_max_num_turns > 1:
            merged = []
            for i in range(0, len(speaker_chunks), chunk_max_num_turns):
                merged.append("\n".join(speaker_chunks[i:i + chunk_max_num_turns]))
            return merged
        return speaker_chunks
    elif chunk_method == "word":
        language = langid.classify(text)[0]
        paragraphs = text.split("\n\n")
        chunks = []
        for paragraph in paragraphs:
            if language == "zh":
                words = list(jieba.cut(paragraph, cut_all=False))
            else:
                words = paragraph.split(" ")
            for i in range(0, len(words), chunk_max_word_num):
                chunk = "".join(words[i:i + chunk_max_word_num]) if language == "zh" else " ".join(words[i:i + chunk_max_word_num])
                chunks.append(chunk)
        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _build_system_message_with_audio_prompt(system_message: str) -> Message:
    contents = []
    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN):]
    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    return Message(role="system", content=contents)


def prepare_generation_context(scene_prompt: Optional[str], ref_audio: Optional[str],
                               ref_audio_in_system_message: bool, audio_tokenizer,
                               speaker_tags: List[str]) -> Tuple[List[Message], List[torch.Tensor]]:
    system_message, messages, audio_ids = None, [], []
    if ref_audio is not None:
        num_speakers = len(ref_audio.split(","))
        speaker_info_l = ref_audio.split(",")
        voice_profile = None
        if any([s.startswith("profile:") for s in speaker_info_l]):
            ref_audio_in_system_message = True
        if ref_audio_in_system_message:
            speaker_desc = []
            for spk_id, character_name in enumerate(speaker_info_l):
                if character_name.startswith("profile:"):
                    if voice_profile is None:
                        with open(f"{CURR_DIR}/voice_prompts/profile.yaml", "r", encoding="utf-8") as f:
                            voice_profile = yaml.safe_load(f)
                    character_desc = voice_profile["profiles"][character_name[len("profile:"):].strip()]
                    speaker_desc.append(f"SPEAKER{spk_id}: {character_desc}")
                else:
                    speaker_desc.append(f"SPEAKER{spk_id}: {AUDIO_PLACEHOLDER_TOKEN}")
            if scene_prompt:
                system_message = ("Generate audio following instruction.\n\n"
                                  f"<|scene_desc_start|>\n{scene_prompt}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>")
            else:
                system_message = ("Generate audio following instruction.\n\n"
                                  "<|scene_desc_start|>\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>")
            system_message = _build_system_message_with_audio_prompt(system_message)
        else:
            if scene_prompt:
                system_message = Message(role="system",
                                         content=f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
        voice_profile = None
        for spk_id, character_name in enumerate(speaker_info_l):
            if not character_name.startswith("profile:"):
                prompt_audio_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name}.wav")
                prompt_text_path = os.path.join(f"{CURR_DIR}/voice_prompts", f"{character_name}.txt")
                assert os.path.exists(prompt_audio_path), f"Voice prompt audio file {prompt_audio_path} does not exist."
                assert os.path.exists(prompt_text_path), f"Voice prompt text file {prompt_text_path} does not exist."
                with open(prompt_text_path, "r", encoding="utf-8") as f:
                    prompt_text = f.read().strip()
                audio_tokens = audio_tokenizer.encode(prompt_audio_path)
                audio_ids.append(audio_tokens)
                if not ref_audio_in_system_message:
                    messages.append(Message(role="user", content=f"[SPEAKER{spk_id}] {prompt_text}" if num_speakers > 1 else prompt_text))
                    messages.append(Message(role="assistant", content=AudioContent(audio_url=prompt_audio_path)))
    else:
        if len(speaker_tags) > 1:
            speaker_desc_l = []
            for idx, tag in enumerate(speaker_tags):
                speaker_desc_l.append(f"{tag}: {'feminine' if idx % 2 == 0 else 'masculine'}")
            speaker_desc = "\n".join(speaker_desc_l)
            scene_desc_l = [scene_prompt] if scene_prompt else []
            scene_desc_l.append(speaker_desc)
            scene_desc_text = "\n\n".join(scene_desc_l)
            system_message = Message(
                role="system",
                content=f"{MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE}\n\n<|scene_desc_start|>\n{scene_desc_text}\n<|scene_desc_end|>"
            )
        else:
            system_message_l = ["Generate audio following instruction."]
            if scene_prompt:
                system_message_l.append(f"<|scene_desc_start|>\n{scene_prompt}\n<|scene_desc_end|>")
            system_message = Message(role="system", content="\n\n".join(system_message_l))
    if system_message:
        messages.insert(0, system_message)
    return messages, audio_ids


def _preprocess_transcript(raw: str) -> str:
    pattern = re.compile(r"\[(SPEAKER\d+)\]")
    speaker_tags = sorted(set(pattern.findall(raw)))
    txt = normalize_chinese_punctuation(raw)
    txt = txt.replace("(", " ").replace(")", " ")
    txt = txt.replace("°F", " degrees Fahrenheit").replace("°C", " degrees Celsius")
    for tag, repl in [
        ("[laugh]", "<SE>[Laughter]</SE>"),
        ("[humming start]", "<SE_s>[Humming]</SE_s>"),
        ("[humming end]", "<SE_e>[Humming]</SE_e>"),
        ("[music start]", "<SE_s>[Music]</SE_s>"),
        ("[music end]", "<SE_e>[Music]</SE_e>"),
        ("[music]", "<SE>[Music]</SE>"),
        ("[sing start]", "<SE_s>[Singing]</SE_s>"),
        ("[sing end]", "<SE_e>[Singing]</SE_e>"),
        ("[applause]", "<SE>[Applause]</SE>"),
        ("[cheering]", "<SE>[Cheering]</SE>"),
        ("[cough]", "<SE>[Cough]</SE>"),
    ]:
        txt = txt.replace(tag, repl)
    lines = txt.split("\n")
    txt = "\n".join([" ".join(line.split()) for line in lines if line.strip()]).strip()
    if not any([txt.endswith(c) for c in [".", "!", "?", ",", ";", '"', "'", "</SE_e>", "</SE>"]]):
        txt += "."
    return txt

# -------------------- Model Client --------------------
class HiggsAudioModelClient:
    def __init__(self, model_path, audio_tokenizer, device=None, device_id=None, max_new_tokens=1024,
                 kv_cache_lengths: List[int] = [4096, 8192], use_static_kv_cache=True):
        if device_id is not None:
            device = f"cuda:{device_id}"
            self._device = device
        else:
            if device is not None:
                self._device = device
            else:
                if torch.cuda.is_available():
                    self._device = "cuda:0"
                elif torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"

        logger.info(f"Using device: {self._device}")
        if isinstance(audio_tokenizer, str):
            audio_tokenizer_device = "cpu" if self._device == "mps" else self._device
            self._audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer, device=audio_tokenizer_device)
        else:
            self._audio_tokenizer = audio_tokenizer

        self._model = HiggsAudioModel.from_pretrained(
            model_path, device_map=self._device, torch_dtype=torch.bfloat16
        )
        self._model.eval()
        self._kv_cache_lengths = kv_cache_lengths
        self._use_static_kv_cache = use_static_kv_cache

        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self._config = AutoConfig.from_pretrained(model_path)
        self._max_new_tokens = max_new_tokens
        self._collator = HiggsAudioSampleCollator(
            whisper_processor=None,
            audio_in_token_id=self._config.audio_in_token_idx,
            audio_out_token_id=self._config.audio_out_token_idx,
            audio_stream_bos_id=self._config.audio_stream_bos_id,
            audio_stream_eos_id=self._config.audio_stream_eos_id,
            encode_whisper_embed=self._config.encode_whisper_embed,
            pad_token_id=self._config.pad_token_id,
            return_audio_in_tokens=self._config.encode_audio_in_tokens,
            use_delay_pattern=self._config.use_delay_pattern,
            round_to=1,
            audio_num_codebooks=self._config.audio_num_codebooks,
        )
        self.kv_caches = None
        if use_static_kv_cache and "cuda" in self._device:
            self._init_static_kv_cache()

    def _init_static_kv_cache(self):
        cache_config = copy.deepcopy(self._model.config.text_config)
        cache_config.num_hidden_layers = self._model.config.text_config.num_hidden_layers
        if self._model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self._model.config.audio_dual_ffn_layers)
        self.kv_caches = {
            length: StaticCache(
                config=cache_config, max_batch_size=8, max_cache_len=length,
                device=self._model.device, dtype=self._model.dtype,
            )
            for length in sorted(self._kv_cache_lengths)
        }
        logger.info(f"Capturing CUDA graphs for KV cache lengths: {sorted(self._kv_cache_lengths)}")
        self._model.capture_model(self.kv_caches.values())

    def _reset_kv_caches(self):
        if self.kv_caches:
            for kv_cache in self.kv_caches.values():
                kv_cache.reset()

    @torch.inference_mode()
    def generate_batched(self, messages_batch: List[List[Message]],
                         audio_ids_batch: List[List[torch.Tensor]],
                         texts_batch: List[str],
                         temperature=0.9, top_k=0, top_p=0.95,
                         ras_win_len=7, ras_win_max_num_repeat=2, seed: Optional[int]=123):
        """True batched generation: one GPU call for N items."""
        postfix = self._tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
        samples = []
        for messages, audio_ids, text in zip(messages_batch, audio_ids_batch, texts_batch):
            chatml_sample = ChatMLSample(messages=messages + [Message(role="user", content=text)])
            input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self._tokenizer)
            input_tokens.extend(postfix)
            curr_sample = ChatMLDatasetSample(
                input_ids=torch.LongTensor(input_tokens),
                label_ids=None,
                audio_ids_concat=torch.concat([ele.cpu() for ele in audio_ids], dim=1) if audio_ids else None,
                audio_ids_start=torch.cumsum(torch.tensor([0] + [ele.shape[1] for ele in audio_ids], dtype=torch.long), dim=0) if audio_ids else None,
                audio_waveforms_concat=None, audio_waveforms_start=None, audio_sample_rate=None, audio_speaker_indices=None,
            )
            samples.append(curr_sample)

        batch_data = self._collator(samples)
        batch = {k: (v.contiguous().to(self._device) if isinstance(v, torch.Tensor) else v)
                 for k, v in asdict(batch_data).items()}

        if self._use_static_kv_cache and self.kv_caches:
            self._reset_kv_caches()

        outputs = self._model.generate(
            **batch,
            max_new_tokens=self._max_new_tokens,
            use_cache=True,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,  # 0 disables top-k; rely on top-p
            top_p=top_p,
            past_key_values_buckets=self.kv_caches,
            ras_win_len=ras_win_len, ras_win_max_num_repeat=ras_win_max_num_repeat,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            tokenizer=self._tokenizer,
            seed=seed,
        )

        # outputs: (decoded_text_ids, audio_codebooks_list_per_sample)
        wavs = []
        sr = 24000
        decoded_texts = outputs[0]
        audio_tokens_list = outputs[1]  # expected list-like of length batch_size
        for b in range(len(texts_batch)):
            audio_out_ids = audio_tokens_list[b]
            if self._config.use_delay_pattern:
                audio_out_ids = revert_delay_pattern(audio_out_ids)
            audio_out_ids = audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1]
            audio_out_ids_cpu = audio_out_ids.detach().cpu() if audio_out_ids.device.type == "mps" else audio_out_ids
            wav = self._audio_tokenizer.decode(audio_out_ids_cpu.unsqueeze(0))[0, 0]
            text_result = self._tokenizer.decode(decoded_texts[b])
            wavs.append((wav, sr, text_result))
        return wavs

# -------------------- Multithreaded Pipeline --------------------
class Item:
    """Represents a single work item through the pipeline."""
    __slots__ = ("idx", "raw_prompt", "voice", "out_path", "script_path", "messages", "audio_ids", "chunks")
    def __init__(self, idx: int, raw_prompt: str, voice: Optional[str],
                 out_path: str, script_path: str):
        self.idx = idx
        self.raw_prompt = raw_prompt
        self.voice = voice
        self.out_path = out_path
        self.script_path = script_path
        self.messages: List[Message] = []
        self.audio_ids: List[torch.Tensor] = []
        self.chunks: List[str] = []  # we will use only first chunk for speed (single pass)

def preprocessor_worker(items: List[Item], scene_prompt_text: Optional[str],
                        ref_audio_in_system_message: bool, audio_tokenizer_obj,
                        chunk_method: Optional[str], chunk_max_word_num: int,
                        chunk_max_num_turns: int, ready_q: queue.Queue, device_str: str):
    """Prepare messages, audio_ids, and first chunk text for each item."""
    for it in items:
        txt = _preprocess_transcript(it.raw_prompt)
        pattern = re.compile(r"\[(SPEAKER\d+)\]")
        speaker_tags = sorted(set(pattern.findall(txt)))
        messages, audio_ids = prepare_generation_context(
            scene_prompt=scene_prompt_text,
            ref_audio=it.voice,
            ref_audio_in_system_message=ref_audio_in_system_message,
            audio_tokenizer=audio_tokenizer_obj,
            speaker_tags=speaker_tags,
        )
        # fast path: we only keep first chunk (for speed). If chunking requested, merge to single.
        if chunk_method is None:
            chunks = [txt]
        else:
            parts = prepare_chunk_text(
                txt, chunk_method=chunk_method,
                chunk_max_word_num=chunk_max_word_num,
                chunk_max_num_turns=chunk_max_num_turns,
            )
            # Merge all chunks to a single text for highest throughput (optional trade-off)
            chunks = [" ".join(parts)]
        it.messages = messages
        it.audio_ids = audio_ids
        it.chunks = chunks
        ready_q.put(it)

def compute_worker(model_client: HiggsAudioModelClient, ready_q: queue.Queue, out_q: queue.Queue,
                   total_items: int, batch_size: int, temperature: float, top_k: int, top_p: float,
                   ras_win_len: int, ras_win_max_num_repeat: int, seed: Optional[int]):
    """Pull items, batch them, run GPU once per batch, push waveforms to out_q."""
    processed = 0
    while processed < total_items:
        batch_items: List[Item] = []
        while len(batch_items) < batch_size and processed + len(batch_items) < total_items:
            try:
                it = ready_q.get(timeout=0.05)
                batch_items.append(it)
            except queue.Empty:
                # if we have at least one, go ahead; else wait a bit more
                if batch_items:
                    break
                else:
                    time.sleep(0.01)

        if not batch_items:
            continue

        messages_batch = [it.messages for it in batch_items]
        audio_ids_batch = [it.audio_ids for it in batch_items]
        texts_batch = [it.chunks[0] for it in batch_items]

        outputs = model_client.generate_batched(
            messages_batch=messages_batch,
            audio_ids_batch=audio_ids_batch,
            texts_batch=texts_batch,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            ras_win_len=ras_win_len,
            ras_win_max_num_repeat=ras_win_max_num_repeat,
            seed=seed,
        )

        for it, (wav, sr, _) in zip(batch_items, outputs):
            out_q.put((it, wav, sr))
        processed += len(batch_items)

def saver_worker(out_q: queue.Queue, io_pool: ThreadPoolExecutor, total_items: int):
    """Consume GPU outputs and dispatch async file writes."""
    saved = 0
    futures = []
    def _save_pair(audio_path, wav, sr, text_path, raw_prompt):
        sf.write(audio_path, wav, sr)
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(raw_prompt)

    while saved < total_items:
        try:
            it, wav, sr = out_q.get(timeout=0.1)
        except queue.Empty:
            continue
        futures.append(io_pool.submit(_save_pair, it.out_path, wav, sr, it.script_path, it.raw_prompt))
        saved += 1

    # finalize writes
    for f in futures:
        f.result()

# -------------------- CLI --------------------
@click.command()
@click.option("--model_path", type=str, default="bosonai/higgs-audio-v2-generation-3B-base")
@click.option("--audio_tokenizer", type=str, default="bosonai/higgs-audio-v2-tokenizer")
@click.option("--max_new_tokens", type=int, default=1024, help="Reduce for shorter clips & higher throughput.")
@click.option("--transcript", type=str, default=None, help="Single text file path (fallback if --texts_file not given).")
@click.option("--texts_file", type=str, default=None,
              help="TXT (one prompt per line) or JSON (array of objects with prompt/out_filename/voice).")
@click.option("--batch_size", type=int, default=5, help="True GPU batch size. Tune for VRAM (A6000 48GB: try 5-8).")
@click.option("--scene_prompt", type=str, default=f"{CURR_DIR}/scene_prompts/quiet_indoor.txt")
@click.option("--temperature", type=float, default=0.9)
@click.option("--top_k", type=int, default=0, help="0 disables top-k; rely on top-p for speed.")
@click.option("--top_p", type=float, default=0.95)
@click.option("--ras_win_len", type=int, default=7)
@click.option("--ras_win_max_num_repeat", type=int, default=2)
@click.option("--ref_audio", type=str, default=None)
@click.option("--ref_audio_in_system_message", is_flag=True, default=False, show_default=True)
@click.option("--chunk_method", default=None, type=click.Choice([None, "speaker", "word"]))
@click.option("--chunk_max_word_num", default=200, type=int)
@click.option("--chunk_max_num_turns", default=1, type=int)
@click.option("--seed", default=123, type=int)
@click.option("--device_id", type=int, default=0)
@click.option("--out_dir", type=str, default="outputs", help="Root directory to write WAVs.")
@click.option("--out_name", type=str, default="generation", help="Default base name if using TXT/TRANSCRIPT.")
@click.option("--use_static_kv_cache", type=int, default=1)
@click.option("--device", type=click.Choice(["auto", "cuda", "mps", "none"]), default="auto")
def main(model_path, audio_tokenizer, max_new_tokens, transcript, texts_file, batch_size, scene_prompt, temperature, top_k, top_p,
         ras_win_len, ras_win_max_num_repeat, ref_audio, ref_audio_in_system_message, chunk_method, chunk_max_word_num,
         chunk_max_num_turns, seed, device_id, out_dir, out_name, use_static_kv_cache, device):

    # ---- Torch perf knobs (speed!) ----
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(1)  # reduce CPU contention in preprocessing

    # ---- Device resolution ----
    if device == "auto":
        if torch.cuda.is_available():
            device = f"cuda:{device_id}"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    elif device == "cuda":
        device = f"cuda:{device_id}"
    elif device == "mps":
        device = "mps"
    else:
        device = "cpu"

    # ---- Scene prompt (optional) ----
    if scene_prompt is not None and scene_prompt != "empty" and os.path.exists(scene_prompt):
        with open(scene_prompt, "r", encoding="utf-8") as f:
            scene_prompt_text = f.read().strip()
    else:
        scene_prompt_text = None

    # ---- Tokenizer/model ONCE ----
    audio_tokenizer_device = "cpu" if device == "mps" else device
    audio_tokenizer_obj = load_higgs_audio_tokenizer(audio_tokenizer, device=audio_tokenizer_device)
    if device == "mps" and use_static_kv_cache:
        use_static_kv_cache = 0

    model_client = HiggsAudioModelClient(
        model_path=model_path,
        audio_tokenizer=audio_tokenizer_obj,
        device=device,
        device_id=None if not device.startswith("cuda") else int(device.split(":")[1]),
        max_new_tokens=max_new_tokens,
        use_static_kv_cache=bool(use_static_kv_cache),
        kv_cache_lengths=[4096, 8192],
    )

    os.makedirs(out_dir, exist_ok=True)

    # ---- Load inputs ----
    texts: List[dict] = []
    if texts_file:
        assert os.path.exists(texts_file), f"{texts_file} not found"
        if texts_file.lower().endswith(".json"):
            with open(texts_file, "r", encoding="utf-8") as f:
                texts = json.load(f)
            assert isinstance(texts, list), "JSON must be an array of objects"
            for item in texts:
                assert "prompt" in item and "out_filename" in item, "Each object must have 'prompt' and 'out_filename'."
        else:
            with open(texts_file, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            texts = [{"prompt": ln, "out_filename": f"{out_name}_{i+1:03d}.wav", "voice": ref_audio} for i, ln in enumerate(lines)]
    elif transcript:
        assert os.path.exists(transcript), f"{transcript} not found"
        with open(transcript, "r", encoding="utf-8") as f:
            texts = [{"prompt": f.read().strip(), "out_filename": f"{out_name}.wav", "voice": ref_audio}]
    else:
        raise ValueError("Provide --texts_file (TXT/JSON) or --transcript (single file).")

    total_items = len(texts)
    logger.info(f"Loaded {total_items} item(s). Target batch_size={batch_size}. Device={device}")

    # ---- Build Item objects & output paths ----
    items: List[Item] = []
    for idx, obj in enumerate(texts):
        raw_prompt = obj["prompt"]
        out_filename_with_path = obj["out_filename"]
        voice = obj.get("voice", ref_audio)

        folder_name, file_name = os.path.split(out_filename_with_path.replace("\\", os.sep))
        full_output_folder_path = os.path.join(out_dir, folder_name)
        os.makedirs(full_output_folder_path, exist_ok=True)

        audio_out_path = os.path.join(full_output_folder_path, file_name)
        text_out_path = os.path.join(full_output_folder_path, "script.txt")
        items.append(Item(idx=idx, raw_prompt=raw_prompt, voice=voice,
                          out_path=audio_out_path, script_path=text_out_path))

    # ---- Multithreaded pipeline queues ----
    ready_q: queue.Queue = queue.Queue(maxsize=max(8, batch_size * 4))
    out_q: queue.Queue = queue.Queue(maxsize=max(8, batch_size * 4))

    # ---- ThreadPool for I/O ----
    io_pool = ThreadPoolExecutor(max_workers=4)

    # ---- Start threads ----
    prep_t = threading.Thread(
        target=preprocessor_worker,
        args=(items, scene_prompt_text, ref_audio_in_system_message, audio_tokenizer_obj,
              chunk_method, chunk_max_word_num, chunk_max_num_turns, ready_q, device),
        daemon=True
    )
    comp_t = threading.Thread(
        target=compute_worker,
        args=(model_client, ready_q, out_q, total_items, batch_size, temperature, top_k, top_p,
              ras_win_len, ras_win_max_num_repeat, seed),
        daemon=True
    )
    save_t = threading.Thread(
        target=saver_worker,
        args=(out_q, io_pool, total_items),
        daemon=True
    )

    t0 = time.time()
    prep_t.start()
    comp_t.start()
    save_t.start()

    # ---- Join threads ----
    prep_t.join()
    comp_t.join()
    save_t.join()
    io_pool.shutdown(wait=True)
    t1 = time.time()

    logger.info(f"All {total_items} item(s) processed successfully in {t1 - t0:.2f}s.")

if __name__ == "__main__":
    main()
