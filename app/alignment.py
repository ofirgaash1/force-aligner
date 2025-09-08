from __future__ import annotations

from typing import List, Dict


def logits_from_audio(model_bundle, waveform, sample_rate: int):
    """
    Run the model forward and return emission logits [T, V].
    """
    import torch  # type: ignore

    processor = model_bundle.processor
    model = model_bundle.model

    with torch.inference_mode():
        inputs = processor(
            waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt"
        )
        input_values = inputs.input_values.to(model.device)
        logits = model(input_values).logits  # [B, T, V]
        emissions = torch.log_softmax(logits, dim=-1)[0].cpu()  # [T, V]
    return emissions


def align_words(
    emissions,
    transcript: str,
    processor,
    duration_sec: float,
) -> List[Dict]:
    """
    Heuristic CTC alignment without external deps:
    - Take argmax path over time
    - Tokenize transcript using CTC tokenizer (char-level tokens)
    - For each non-delimiter token, find its span on the path
    - Group contiguous characters into words using tokenizer's delimiter
    """
    import numpy as np  # type: ignore

    tokenizer = processor.tokenizer
    word_delim = getattr(tokenizer, "word_delimiter_token", "|")

    # Tokenize transcript to CTC token strings (usually char-level)
    token_strs: List[str] = tokenizer.tokenize(transcript)
    token_ids: List[int] = tokenizer.convert_tokens_to_ids(token_strs)

    # Build mask for delimiter tokens to split words later
    is_delim = [ts == word_delim for ts in token_strs]

    # Argmax path over emissions [T, V]
    emissions_np = emissions.detach().cpu().numpy()
    path_ids = np.argmax(emissions_np, axis=-1)  # [T]
    T = path_ids.shape[0]
    frame_to_sec = duration_sec / float(T)

    # Precompute per-frame max prob for confidence
    frame_max_prob = np.exp(np.max(emissions_np, axis=-1))  # [T]

    # For each target token, find a best-effort [start, end] in frames
    spans: List[tuple[int, int]] = []
    cursor = 0
    for tid in token_ids:
        if tid is None:
            spans.append((cursor, cursor))
            continue
        # Search forward for first occurrence
        start = -1
        last_seen = -1
        for t in range(cursor, T):
            if path_ids[t] == tid:
                if start == -1:
                    start = t
                last_seen = t
            elif start != -1 and last_seen != -1 and t - last_seen > 4:
                # Small gap tolerance; assume token ended
                break
        if start == -1:
            # Not found: approximate by placing zero-length span at cursor
            start = cursor
            end = cursor
        else:
            end = last_seen if last_seen != -1 else start
        spans.append((start, max(start, end)))
        cursor = max(cursor, end)

    # Aggregate character spans into words
    words: List[Dict] = []
    current_chars: List[int] = []
    current_text: List[str] = []

    def flush_word():
        if not current_chars:
            return
        s_frame = spans[current_chars[0]][0]
        e_frame = spans[current_chars[-1]][1]
        s_sec = max(0.0, s_frame * frame_to_sec)
        e_sec = max(s_sec, e_frame * frame_to_sec)
        # Confidence: average frame max prob across [s_frame, e_frame]
        if e_frame >= s_frame:
            conf = float(np.mean(frame_max_prob[s_frame : e_frame + 1]))
        else:
            conf = 0.0
        words.append(
            {
                "word": "".join(current_text).replace(word_delim, " "),
                "start": s_sec,
                "end": e_sec,
                "score": max(0.0, min(1.0, conf)),
            }
        )

    for i, (ts, is_d) in enumerate(zip(token_strs, is_delim)):
        if is_d:
            flush_word()
            current_chars = []
            current_text = []
            continue
        current_chars.append(i)
        current_text.append(ts)

    flush_word()

    # Filter empty words (can occur with leading/trailing delimiter)
    words = [w for w in words if w["word"].strip()]
    return words
