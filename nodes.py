import os
import sys
import torch
import torchaudio
import numpy as np
import uuid
import gc
import folder_paths
import json
from dataclasses import dataclass
from tqdm import tqdm

# ----------------------------
# Add Local HeartLib to Path
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
util_dir = os.path.join(current_dir, "util")
if util_dir not in sys.path:
    sys.path.insert(0, util_dir)

from heartlib.heartmula.modeling_heartmula import HeartMuLa
from heartlib.heartcodec.modeling_heartcodec import HeartCodec
from tokenizers import Tokenizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
from huggingface_hub import snapshot_download

# ----------------------------
# Configuration Class
# ----------------------------
@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str):
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return cls(**data)

# ----------------------------
# Helper Functions
# ----------------------------
def _cfg_cat(tensor: torch.Tensor, cfg_scale: float):
    tensor = tensor.unsqueeze(0)
    if cfg_scale != 1.0:
        tensor = torch.cat([tensor, tensor], dim=0)
    return tensor

def download_model_if_needed(repo_id, local_dir):
    if not os.path.exists(local_dir):
        print(f"[HeartMuLa] Downloading {repo_id} to {local_dir}...")
        snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
        print(f"[HeartMuLa] Downloaded {repo_id}.")

# ----------------------------
# Node: HeartMuLa Model Loader
# ----------------------------
class HeartMuLaModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "version": (["3B", "7B"], {"default": "3B"}),
                "load_device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("HEARTMULA_MODEL",)
    RETURN_NAMES = ("heartmula_model",)
    FUNCTION = "load_model"
    CATEGORY = "HeartMuLa"

    def load_model(self, version, load_device):
        model_base_dir = os.path.join(folder_paths.models_dir, "HeartMuLa")
        model_path = os.path.join(model_base_dir, f"HeartMuLa-oss-{version}")
        
        # Auto Download
        if not os.path.exists(model_path):
            print(f"[HeartMuLa] Model not found at {model_path}. Downloading...")
            # Download base config files if missing
            if not os.path.exists(os.path.join(model_base_dir, "gen_config.json")):
                 download_model_if_needed("HeartMuLa/HeartMuLaGen", model_base_dir)
            
            # Download specific model version
            download_model_if_needed(f"HeartMuLa/HeartMuLa-oss-{version}", model_path)
        
        print(f"[HeartMuLa] Loading HeartMuLa Model ({version}) to {load_device}...")
        
        # Load Config
        gen_config_path = os.path.join(model_path, "gen_config.json")
        if not os.path.exists(gen_config_path):
             # Try parent dir if not in version dir (structure in Readme says gen_config.json is in ckpt root)
             gen_config_path = os.path.join(model_base_dir, "gen_config.json")
        
        if not os.path.exists(gen_config_path):
             raise FileNotFoundError(f"gen_config.json not found in {model_path} or {model_base_dir}")

        config = HeartMuLaGenConfig.from_file(gen_config_path)

        # Load Tokenizer
        tokenizer_path = os.path.join(model_path, "tokenizer.json")
        if not os.path.exists(tokenizer_path):
             tokenizer_path = os.path.join(model_base_dir, "tokenizer.json")
        
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"tokenizer.json not found.")

        tokenizer = Tokenizer.from_file(tokenizer_path)

        # Load Model
        # Using bfloat16 as default for these models
        model = HeartMuLa.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        
        if load_device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")
        
        heartmula_model = {
            "model": model,
            "tokenizer": tokenizer,
            "config": config,
            "version": version
        }
        print(f"[HeartMuLa] Model ({version}) loaded on {load_device}.")
        return (heartmula_model,)

# ----------------------------
# Node: HeartCodec Loader
# ----------------------------
class HeartCodecLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Default path usually inside HeartMuLa folder or separate
                "model_name": (["HeartCodec-oss"], {"default": "HeartCodec-oss"}),
                "load_device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("HEARTCODEC_MODEL",)
    RETURN_NAMES = ("heartcodec_model",)
    FUNCTION = "load_codec"
    CATEGORY = "HeartMuLa"

    def load_codec(self, model_name, load_device):
        model_base_dir = os.path.join(folder_paths.models_dir, "HeartMuLa")
        codec_path = os.path.join(model_base_dir, model_name)
        
        # Auto Download
        if not os.path.exists(codec_path):
            print(f"[HeartMuLa] HeartCodec not found at {codec_path}. Downloading...")
            download_model_if_needed(f"HeartMuLa/{model_name}", codec_path)

        print(f"[HeartMuLa] Loading HeartCodec ({model_name}) to {load_device}...")
        
        # Load Codec
        device = torch.device(load_device if torch.cuda.is_available() and load_device == "cuda" else "cpu")
        codec = HeartCodec.from_pretrained(codec_path, device_map=device)
        
        heartcodec_model = {
            "model": codec,
            "name": model_name
        }
        print(f"[HeartMuLa] HeartCodec loaded on {load_device}.")
        return (heartcodec_model,)

# ----------------------------
# Node: HeartMuLa Generator
# ----------------------------
class HeartMuLaGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "heartmula_model": ("HEARTMULA_MODEL",),
                "heartcodec_model": ("HEARTCODEC_MODEL",),
                "lyrics": ("STRING", {"multiline": True, "placeholder": "[Verse]\n..."}),
                "tags": ("STRING", {"multiline": True, "placeholder": "piano,happy,wedding"}),
                "max_audio_length_ms": ("INT", {"default": 60000, "min": 10000, "max": 600000, "step": 10000}),
                "topk": ("INT", {"default": 50, "min": 1, "max": 200, "step": 1}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "cfg_scale": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "control_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "low_vram": ("BOOLEAN", {"default": True, "label": "Low VRAM Mode (Offload models)"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("audio_output", "filepath")
    FUNCTION = "generate"
    CATEGORY = "HeartMuLa"

    def generate(self, heartmula_model, heartcodec_model, lyrics, tags, max_audio_length_ms, topk, temperature, cfg_scale, seed, control_seed, low_vram):
        # Setup Device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[HeartMuLa] Generation started on {device}...")

        # Unpack Models
        model = heartmula_model["model"]
        tokenizer = heartmula_model["tokenizer"]
        config = heartmula_model["config"]
        codec = heartcodec_model["model"]
        
        # --- SEED HANDLING ---
        if seed > 0:
            torch.manual_seed(seed)
        elif control_seed > 0:
            torch.manual_seed(control_seed)

        # --- PREPROCESS (Logic ported from HeartMuLaGenPipeline) ---
        tags = tags.lower()
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"

        tags_ids = tokenizer.encode(tags).ids
        if tags_ids[0] != config.text_bos_id:
            tags_ids = [config.text_bos_id] + tags_ids
        if tags_ids[-1] != config.text_eos_id:
            tags_ids = tags_ids + [config.text_eos_id]

        muq_dim = model.config.muq_dim
        dtype = model.dtype
        muq_embed = torch.zeros([muq_dim], dtype=dtype) # Will be moved to device later
        muq_idx = len(tags_ids)

        # process lyrics
        lyrics = lyrics.lower()
        lyrics_ids = tokenizer.encode(lyrics).ids
        if lyrics_ids[0] != config.text_bos_id:
            lyrics_ids = [config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != config.text_eos_id:
            lyrics_ids = lyrics_ids + [config.text_eos_id]

        # cat them together
        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
        parallel_number = codec.config.num_quantizers + 1
        
        tokens = torch.zeros([prompt_len, parallel_number], dtype=torch.long)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1

        # Prepare Tensors (Keep on CPU until model move)
        tokens = _cfg_cat(tokens, cfg_scale)
        tokens_mask = _cfg_cat(tokens_mask, cfg_scale)
        muq_embed = _cfg_cat(muq_embed, cfg_scale)
        muq_idx_list = [muq_idx] * bs_size
        pos = _cfg_cat(torch.arange(prompt_len, dtype=torch.long), cfg_scale)

        # --- STEP 1: GENERATE TOKENS (HeartMuLa) ---
        print("[HeartMuLa] Moving HeartMuLa model to GPU...")
        model.to(device)
        
        # Move inputs to device
        tokens = tokens.to(device)
        tokens_mask = tokens_mask.to(device)
        muq_embed = muq_embed.to(device)
        pos = pos.to(device)

        frames = []
        
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            model.setup_caches(bs_size)
            curr_token = model.generate_frame(
                tokens=tokens,
                tokens_mask=tokens_mask,
                input_pos=pos,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=muq_embed,
                starts=muq_idx_list,
            )
        
        frames.append(curr_token[0:1,])

        def _pad_audio_token(token: torch.Tensor):
            padded_token = (
                torch.ones(
                    (token.shape[0], parallel_number),
                    device=token.device,
                    dtype=torch.long,
                )
                * config.empty_id
            )
            padded_token[:, :-1] = token
            padded_token = padded_token.unsqueeze(1)
            padded_token_mask = torch.ones_like(
                padded_token, device=token.device, dtype=torch.bool
            )
            padded_token_mask[..., -1] = False
            return padded_token, padded_token_mask

        max_audio_frames = max_audio_length_ms // 80
        
        print(f"[HeartMuLa] Generating {max_audio_frames} frames...")
        for i in tqdm(range(max_audio_frames)):
            curr_token, curr_token_mask = _pad_audio_token(curr_token)
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
                curr_token = model.generate_frame(
                    tokens=curr_token,
                    tokens_mask=curr_token_mask,
                    input_pos=pos[..., -1:] + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=None,
                    starts=None,
                )
            if torch.any(curr_token[0:1, :] >= config.audio_eos_id):
                break
            frames.append(curr_token[0:1,])
        
        # Concatenate frames
        frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
        
        # Offload HeartMuLa if needed
        if low_vram:
            print("[HeartMuLa] Offloading HeartMuLa model to CPU...")
            model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

        # --- STEP 2: DECODE AUDIO (HeartCodec) ---
        print("[HeartMuLa] Moving HeartCodec model to GPU...")
        codec.to(device)
        frames = frames.to(device)

        print("[HeartMuLa] Decoding audio...")
        with torch.no_grad():
            wav = codec.detokenize(frames)

        # Offload HeartCodec if needed
        if low_vram:
            print("[HeartMuLa] Offloading HeartCodec model to CPU...")
            codec.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

        # Save and Return
        output_dir = folder_paths.get_output_directory()
        os.makedirs(output_dir, exist_ok=True)
        filename = f"heartmula_gen_{uuid.uuid4().hex}.mp3"
        out_path = os.path.join(output_dir, filename)
        
        # Ensure wav is (Channels, Time) for saving
        # detokenize output shape is typically (1, T) or (T)
        wav_cpu = wav.cpu()
        if wav_cpu.ndim == 1:
            wav_cpu = wav_cpu.unsqueeze(0) # (1, T)
        
        torchaudio.save(out_path, wav_cpu, 48000)

        # For ComfyUI AUDIO type: (1, C, T) batch or (C, T)
        # ComfyUI typically expects (Batch, Channels, Time) or just (Channels, Time)
        # Let's check standard. Usually (1, C, T) is safest for single batch.
        
        waveform = wav_cpu
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(0) # (1, C, T)
        
        audio_output = {
            "waveform": waveform,
            "sample_rate": 48000
        }

        return (audio_output, out_path)

# ----------------------------
# Node: HeartTranscriptor Loader
# ----------------------------
class HeartTranscriptorLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (["HeartTranscriptor-oss"], {"default": "HeartTranscriptor-oss"}),
                "load_device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }

    RETURN_TYPES = ("HEART_TRANSCRIPTOR_MODEL",)
    RETURN_NAMES = ("transcriptor_model",)
    FUNCTION = "load_model"
    CATEGORY = "HeartMuLa"

    def load_model(self, model_name, load_device):
        model_base_dir = os.path.join(folder_paths.models_dir, "HeartMuLa")
        model_path = os.path.join(model_base_dir, model_name)
        
        # Auto Download
        if not os.path.exists(model_path):
            print(f"[HeartMuLa] HeartTranscriptor not found at {model_path}. Downloading...")
            download_model_if_needed(f"HeartMuLa/{model_name}", model_path)

        print(f"[HeartMuLa] Loading HeartTranscriptor ({model_name}) to {load_device}...")
        
        # Load Model
        # We load model and processor separately to manage device manually
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = WhisperForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=dtype, 
            low_cpu_mem_usage=True
        )
        
        if load_device == "cuda" and torch.cuda.is_available():
            model = model.to("cuda")

        processor = WhisperProcessor.from_pretrained(model_path)
        
        transcriptor_model = {
            "model": model,
            "processor": processor,
            "dtype": dtype
        }
        print(f"[HeartMuLa] HeartTranscriptor loaded on {load_device}.")
        return (transcriptor_model,)

# ----------------------------
# Node: HeartTranscriptor Runner
# ----------------------------
class HeartTranscriptor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transcriptor_model": ("HEART_TRANSCRIPTOR_MODEL",),
                "audio_input": ("AUDIO",),
                "max_new_tokens": ("INT", {"default": 400, "min": 1, "max": 445, "step": 1}),
                "language": (["auto", "en", "zh", "ja", "ko", "es", "fr", "de"], {"default": "auto"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.1}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1}),
                "chunk_length_s": ("INT", {"default": 30, "min": 15, "max": 30, "step": 1}),
                "stride_length_s": ("INT", {"default": 5, "min": 0, "max": 10, "step": 1}),
                "no_speech_threshold": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05}),
                "logprob_threshold": ("FLOAT", {"default": -1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "low_vram": ("BOOLEAN", {"default": True, "label": "Low VRAM Mode (Offload model)"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lyrics_text",)
    FUNCTION = "transcribe"
    CATEGORY = "HeartMuLa"

    def transcribe(self, transcriptor_model, audio_input, max_new_tokens, language, temperature, batch_size, chunk_length_s, stride_length_s, no_speech_threshold, logprob_threshold, low_vram):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[HeartMuLa] Transcription started on {device}...")

        model = transcriptor_model["model"]
        processor = transcriptor_model["processor"]
        dtype = transcriptor_model["dtype"]

        # --- PRE-PROCESSING ---
        if isinstance(audio_input, dict):
            waveform = audio_input["waveform"]
            sr = audio_input["sample_rate"]
        else:
            sr, waveform = audio_input
        
        # Ensure tensor and float32
        if not isinstance(waveform, torch.Tensor):
             if isinstance(waveform, np.ndarray):
                 waveform = torch.from_numpy(waveform)
             else:
                 waveform = torch.tensor(waveform)
        
        waveform = waveform.float()

        # Normalize shape to (Batch, Channels, Time)
        if waveform.ndim == 1: # (Time)
            waveform = waveform.unsqueeze(0).unsqueeze(0) # -> (1, 1, Time)
        elif waveform.ndim == 2: # (Channels, Time) or (Batch, Time)??
            waveform = waveform.unsqueeze(0) # -> (1, C, T) or (1, B, T)
        
        # Resample if needed (Whisper expects 16000Hz)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
            sr = 16000

        # Mix to Mono: (Batch, Channels, Time) -> (Batch, Time)
        waveform = waveform.mean(dim=1, keepdim=False)

        # Move model to GPU
        print("[HeartMuLa] Moving Transcriptor model to GPU...")
        model.to(device)

        # Prepare numpy input for pipeline
        waveform_np = waveform.cpu().numpy() # (Batch, Time)
        
        # Initialize Pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
            chunk_length_s=chunk_length_s, # Enable chunking for long audio
            stride_length_s=stride_length_s, # Overlap chunks to handle silence/boundaries better
        )

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "no_speech_threshold": no_speech_threshold,
            "logprob_threshold": logprob_threshold,
            "compression_ratio_threshold": 1.8,
            "num_beams": 2,
            "condition_on_prev_tokens": False,
            "task": "transcribe"
        }
        
        if language != "auto":
            gen_kwargs["language"] = language
            
        # Run Inference
        print("[HeartMuLa] Transcribing...")
        
        # Pipeline expects list of numpy arrays for batch, or single array
        if waveform_np.shape[0] == 1:
            inputs = waveform_np[0]
        else:
            inputs = [waveform_np[i] for i in range(waveform_np.shape[0])]

        result = pipe(
            inputs,
            return_timestamps=True, # Enable timestamps internally for better chunking accuracy
            generate_kwargs=gen_kwargs,
            batch_size=batch_size # Process chunks in batches
        )
        
        # Format Output
        final_text = ""
        
        # Handle Batch Result
        if isinstance(result, list):
            results = result
        else:
            results = [result]

        for res in results:
            final_text += res["text"].strip() + "\n"
        
        final_text = final_text.strip()

        # Offload if needed
        if low_vram:
            print("[HeartMuLa] Offloading Transcriptor model to CPU...")
            model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

        return (final_text,)
