# -*- coding: utf-8 -*-
"""Example script for generating multiple voiceovers using HiggsAudio (model loaded once)."""

import click
import soundfile as sf
import langid
import jieba
import os
import re
import copy
import json
import torchaudio
import tqdm
import yaml
import math

from loguru import logger
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import Message, ChatMLSample, AudioContent, TextContent

from boson_multimodal.model.higgs_audio import HiggsAudioConfig, HiggsAudioModel
from boson_multimodal.data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
from boson_multimodal.dataset.chatml_dataset import ChatMLDatasetSample, prepare_chatml_sample
from boson_multimodal.model.higgs_audio.utils import revert_delay_pattern
from typing import List, Optional
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import StaticCache
from dataclasses import asdict
import torch

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PLACEHOLDER_TOKEN = "<|__AUDIO_PLACEHOLDER__|>"

MULTISPEAKER_DEFAULT_SYSTEM_MESSAGE = """You are an AI assistant designed to convert text into speech.
If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.
If no speaker tag is present, select a suitable voice on your own."""


def normalize_chinese_punctuation(text):
    mapping = {
        "：": ":", "；": ";", "？": "?", "！": "!", "（": "(", "）": ")",
        "【": "[", "】": "]", "《": "<", "》": ">", "“": '"', "”": '"', "‘": "'", "’": "'",
        "、": ",", "—": "-", "…": "...", "·": ".", "「": '"', "」": '"', "『": '"', "』": '"',
    }
    # Add space after certain punctuation for better separation
    mapping["，"] = ", "
    mapping["。"] = ". "
    for zh, en in mapping.items():
        text = text.replace(zh, en)
    return text


def prepare_chunk_text(text, chunk_method: Optional[str] = None, chunk_max_word_num: int = 100, chunk_max_num_turns: int = 1):
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
            
            # Handle the last chunk without appending an extra newline
            for i in range(0, len(words), chunk_max_word_num):
                chunk = "".join(words[i:i + chunk_max_word_num]) if language == "zh" else " ".join(words[i:i + chunk_max_word_num])
                chunks.append(chunk)

        return chunks
    else:
        raise ValueError(f"Unknown chunk method: {chunk_method}")


def _build_system_message_with_audio_prompt(system_message):
    contents = []
    while AUDIO_PLACEHOLDER_TOKEN in system_message:
        loc = system_message.find(AUDIO_PLACEHOLDER_TOKEN)
        contents.append(TextContent(system_message[:loc]))
        contents.append(AudioContent(audio_url=""))
        system_message = system_message[loc + len(AUDIO_PLACEHOLDER_TOKEN):]
    if len(system_message) > 0:
        contents.append(TextContent(system_message))
    return Message(role="system", content=contents)


class HiggsAudioModelClient:
    def __init__(self, model_path, audio_tokenizer, device=None, device_id=None, max_new_tokens=2048,
                 kv_cache_lengths: List[int] = [1024, 4096, 8192], use_static_kv_cache=False):
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
        if use_static_kv_cache:
            self._init_static_kv_cache()

    def _init_static_kv_cache(self):
        cache_config = copy.deepcopy(self._model.config.text_config)
        cache_config.num_hidden_layers = self._model.config.text_config.num_hidden_layers
        if self._model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self._model.config.audio_dual_ffn_layers)
        self.kv_caches = {
            length: StaticCache(
                config=cache_config, max_batch_size=1, max_cache_len=length,
                device=self._model.device, dtype=self._model.dtype,
            )
            for length in sorted(self._kv_cache_lengths)
        }
        if "cuda" in self._device:
            logger.info("Capturing CUDA graphs for each KV cache length")
            self._model.capture_model(self.kv_caches.values())

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    @torch.inference_mode()
    def generate(self, messages_batch, audio_ids_batch, chunked_text_batch, generation_chunk_buffer_size,
                 temperature=1.0, top_k=50, top_p=0.95, ras_win_len=7, ras_win_max_num_repeat=2, seed=123, *args, **kwargs):
        """Generates voiceovers for a batch of inputs."""
        sr = 24000
        output_data = []

        # Process each item in the batch
        for messages, audio_ids, chunked_text in zip(messages_batch, audio_ids_batch, chunked_text_batch):
            audio_out_ids_l, generated_audio_ids, generation_messages = [], [], []
            for idx, chunk_text in tqdm.tqdm(enumerate(chunked_text), desc="Generating audio chunks", total=len(chunked_text)):
                generation_messages.append(Message(role="user", content=chunk_text))
                chatml_sample = ChatMLSample(messages=messages + generation_messages)
                input_tokens, _, _, _ = prepare_chatml_sample(chatml_sample, self._tokenizer)
                postfix = self._tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n", add_special_tokens=False)
                input_tokens.extend(postfix)

                context_audio_ids = audio_ids + generated_audio_ids

                curr_sample = ChatMLDatasetSample(
                    input_ids=torch.LongTensor(input_tokens),
                    label_ids=None,
                    audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1) if context_audio_ids else None,
                    audio_ids_start=torch.cumsum(torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0) if context_audio_ids else None,
                    audio_waveforms_concat=None, audio_waveforms_start=None, audio_sample_rate=None, audio_speaker_indices=None,
                )

                batch_data = self._collator([curr_sample])
                batch = asdict(batch_data)
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.contiguous().to(self._device)

                if self._use_static_kv_cache:
                    self._prepare_kv_caches()

                outputs = self._model.generate(
                    **batch, max_new_tokens=self._max_new_tokens, use_cache=True, do_sample=True,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    past_key_values_buckets=self.kv_caches,
                    ras_win_len=ras_win_len, ras_win_max_num_repeat=ras_win_max_num_repeat,
                    stop_strings=["<|end_of_text|>", "<|eot_id|>"], tokenizer=self._tokenizer, seed=seed,
                )

                step_audio_out_ids_l = []
                for ele in outputs[1]:
                    audio_out_ids = ele
                    if self._config.use_delay_pattern:
                        audio_out_ids = revert_delay_pattern(audio_out_ids)
                    step_audio_out_ids_l.append(audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1])
                audio_out_ids = torch.concat(step_audio_out_ids_l, dim=1)
                audio_out_ids_l.append(audio_out_ids)
                generated_audio_ids.append(audio_out_ids)

                generation_messages.append(Message(role="assistant", content=AudioContent(audio_url="")))
                if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
                    generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
                    generation_messages = generation_messages[(-2 * generation_chunk_buffer_size):]

            concat_audio_out_ids = torch.concat(audio_out_ids_l, dim=1)
            if concat_audio_out_ids.device.type == "mps":
                concat_audio_out_ids_cpu = concat_audio_out_ids.detach().cpu()
            else:
                concat_audio_out_ids_cpu = concat_audio_out_ids

            concat_wv = self._audio_tokenizer.decode(concat_audio_out_ids_cpu.unsqueeze(0))[0, 0]
            text_result = self._tokenizer.decode(outputs[0][0])
            output_data.append((concat_wv, sr, text_result))
            
        return output_data

def prepare_generation_context(scene_prompt, ref_audio, ref_audio_in_system_message, audio_tokenizer, speaker_tags):
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


@click.command()
@click.option("--model_path", type=str, default="bosonai/higgs-audio-v2-generation-3B-base")
@click.option("--audio_tokenizer", type=str, default="bosonai/higgs-audio-v2-tokenizer")
@click.option("--max_new_tokens", type=int, default=2048)
@click.option("--transcript", type=str, default=None, help="Single text file path (fallback if --texts_file not given).")
@click.option("--texts_file", type=str, default=None,
              help="TXT (one prompt per line) or JSON (array of objects) with multiple texts.")
@click.option("--batch_size", type=int, default=4, help="Number of scripts to process simultaneously in a batch.")
@click.option("--scene_prompt", type=str, default=f"{CURR_DIR}/scene_prompts/quiet_indoor.txt")
@click.option("--temperature", type=float, default=1.0)
@click.option("--top_k", type=int, default=50)
@click.option("--top_p", type=float, default=0.95)
@click.option("--ras_win_len", type=int, default=7)
@click.option("--ras_win_max_num_repeat", type=int, default=2)
@click.option("--ref_audio", type=str, default=None)
@click.option("--ref_audio_in_system_message", is_flag=True, default=False, show_default=True)
@click.option("--chunk_method", default=None, type=click.Choice([None, "speaker", "word"]))
@click.option("--chunk_max_word_num", default=200, type=int)
@click.option("--chunk_max_num_turns", default=1, type=int)
@click.option("--generation_chunk_buffer_size", default=None, type=int)
@click.option("--seed", default=None, type=int)
@click.option("--device_id", type=int, default=None)
@click.option("--out_dir", type=str, default="outputs", help="Directory to write all WAVs.")
@click.option("--out_name", type=str, default="generation", help="Base filename, e.g., generation_001.wav")
@click.option("--use_static_kv_cache", type=int, default=1)
@click.option("--device", type=click.Choice(["auto", "cuda", "mps", "none"]), default="auto")
def main(model_path, audio_tokenizer, max_new_tokens, transcript, texts_file, batch_size, scene_prompt, temperature, top_k, top_p,
         ras_win_len, ras_win_max_num_repeat, ref_audio, ref_audio_in_system_message, chunk_method, chunk_max_word_num,
         chunk_max_num_turns, generation_chunk_buffer_size, seed, device_id, out_dir, out_name, use_static_kv_cache, device):

    # Device resolution
    if device_id is None:
        if device == "auto":
            if torch.cuda.is_available():
                device_id = 0; device = "cuda:0"
            elif torch.backends.mps.is_available():
                device_id = None; device = "mps"
            else:
                device_id = None; device = "cpu"
        elif device == "cuda":
            device_id = 0; device = "cuda:0"
        elif device == "mps":
            device_id = None; device = "mps"
        else:
            device_id = None; device = "cpu"
    else:
        device = f"cuda:{device_id}"

    # Scene prompt file optional
    if scene_prompt is not None and scene_prompt != "empty" and os.path.exists(scene_prompt):
        with open(scene_prompt, "r", encoding="utf-8") as f:
            scene_prompt_text = f.read().strip()
    else:
        scene_prompt_text = None

    # Load tokenizer/model ONCE
    audio_tokenizer_device = "cpu" if device == "mps" else device
    audio_tokenizer_obj = load_higgs_audio_tokenizer(audio_tokenizer, device=audio_tokenizer_device)
    if device == "mps" and use_static_kv_cache:
        use_static_kv_cache = False

    model_client = HiggsAudioModelClient(
        model_path=model_path,
        audio_tokenizer=audio_tokenizer_obj,
        device=device,
        device_id=device_id,
        max_new_tokens=max_new_tokens,
        use_static_kv_cache=use_static_kv_cache,
    )

    os.makedirs(out_dir, exist_ok=True)

    texts: List[dict] = []
    if texts_file:
        assert os.path.exists(texts_file), f"{texts_file} not found"
        if texts_file.lower().endswith(".json"):
            with open(texts_file, "r", encoding="utf-8") as f:
                texts = json.load(f)
            assert isinstance(texts, list), "JSON must be an array of objects"
            for item in texts:
                assert "prompt" in item and "out_filename" in item, "Each object in JSON must have 'prompt' and 'out_filename' keys."
        else: # For .txt files
            with open(texts_file, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
            # Each line in TXT file is converted to a dictionary with default values
            texts = [{"prompt": ln, "out_filename": f"{out_name}_{i+1:03d}", "voice": ref_audio} for i, ln in enumerate(lines)]

    elif transcript: # For a single transcript file
        assert os.path.exists(transcript), f"{transcript} not found"
        with open(transcript, "r", encoding="utf-8") as f:
            # Single transcript is also wrapped in a dictionary for consistent processing
            texts = [{"prompt": f.read().strip(), "out_filename": out_name, "voice": ref_audio}]
    else:
        raise ValueError("Provide --texts_file (TXT/JSON) or --transcript (single file).")

    logger.info(f"Loaded {len(texts)} item(s) from input. Processing in batches of {batch_size}.")

    # =========================================================================
    # START OF MODIFIED CODE BLOCK FOR BATCH PROCESSING
    # =========================================================================
    # Process items in batches
    num_items = len(texts)
    num_batches = math.ceil(num_items / batch_size)
    
    for i in tqdm.tqdm(range(num_batches), desc="Processing Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_items)
        batch_items = texts[start_idx:end_idx]
        
        # Prepare inputs for the entire batch
        messages_batch, audio_ids_batch, chunked_text_batch = [], [], []
        for item in batch_items:
            raw_text = item["prompt"]
            ref_audio_for_item = item.get("voice", ref_audio)
            
            txt = _preprocess_transcript(raw_text)
            pattern = re.compile(r"\[(SPEAKER\d+)\]")
            speaker_tags = sorted(set(pattern.findall(txt)))
            
            messages, audio_ids = prepare_generation_context(
                scene_prompt=scene_prompt_text,
                ref_audio=ref_audio_for_item,
                ref_audio_in_system_message=ref_audio_in_system_message,
                audio_tokenizer=audio_tokenizer_obj,
                speaker_tags=speaker_tags,
            )
            
            chunked_text = prepare_chunk_text(
                txt, chunk_method=chunk_method,
                chunk_max_word_num=chunk_max_word_num,
                chunk_max_num_turns=chunk_max_num_turns,
            )
            
            messages_batch.append(messages)
            audio_ids_batch.append(audio_ids)
            chunked_text_batch.append(chunked_text)
            
        # Generate voiceovers for the entire batch
        output_data = model_client.generate(
            messages_batch=messages_batch,
            audio_ids_batch=audio_ids_batch,
            chunked_text_batch=chunked_text_batch,
            generation_chunk_buffer_size=generation_chunk_buffer_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            ras_win_len=ras_win_len,
            ras_win_max_num_repeat=ras_win_max_num_repeat,
            seed=seed,
        )
        
        # Save each generated audio file from the batch
        for j, (concat_wv, sr, _) in enumerate(output_data):
            out_file_name = batch_items[j]["out_filename"]
            out_path = os.path.join(out_dir, f"{out_file_name}.wav")
            sf.write(out_path, concat_wv, sr)
            logger.info(f"Saved: {out_path} (sr={sr})")

    logger.info("All items processed successfully.")
    # =========================================================================
    # END OF MODIFIED CODE BLOCK
    # =========================================================================

if __name__ == "__main__":
    main()
