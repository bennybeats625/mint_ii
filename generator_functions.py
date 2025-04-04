import torch
import torch.nn.functional as F

# token offsets
BEAT_CNT_OFFSET = 1
POSITION_OFFSET = 9
INTERVAL_OFFSET = 21
DURATION_OFFSET = 93
TRACK_END_TOKEN = 149

def get_valid_tokens(sequence):
    if not sequence:
        return [["Beat", BEAT_CNT_OFFSET, POSITION_OFFSET - 1]]  # Start with beat tokens

    last_token = sequence[-1]
    valid_tokens = []

    # **Beat Tokens**
    if BEAT_CNT_OFFSET <= last_token < POSITION_OFFSET:
        valid_tokens.append(["Position", POSITION_OFFSET, INTERVAL_OFFSET - 1])  # Position follows any beat

    # **Position Tokens**
    elif POSITION_OFFSET <= last_token < INTERVAL_OFFSET:
        valid_tokens.append(["Interval", INTERVAL_OFFSET, DURATION_OFFSET - 1])  # Interval follows position

    # **Interval Tokens**
    elif INTERVAL_OFFSET <= last_token < DURATION_OFFSET:
        valid_tokens.append(["Duration", DURATION_OFFSET, TRACK_END_TOKEN - 1])  # Duration follows interval

    # **Duration Tokens**
    elif DURATION_OFFSET <= last_token:
        valid_tokens.append(["Beat", BEAT_CNT_OFFSET, POSITION_OFFSET - 1])  # Start new beat
        valid_tokens.append(["Position", POSITION_OFFSET, INTERVAL_OFFSET - 1])  # Allow next position in beat
        valid_tokens.append(["Interval", INTERVAL_OFFSET, DURATION_OFFSET - 1])  # Allow another interval if stacking

    return valid_tokens

def apply_temperature(probs, temperature):
    """Apply temperature scaling."""
    return torch.multinomial(probs / temperature, 1).item()

def apply_top_k(probs, k):
    """Apply top-k filtering."""
    k_usable = min(k, probs.size(-1))
    top_k_probs, top_k_indices = torch.topk(probs, k_usable)
    sampled_idx = torch.multinomial(top_k_probs.squeeze(0), 1).item()
    return top_k_indices[0, sampled_idx].item()

def apply_top_p(probs, p):
    """Apply nucleus (top-p) sampling."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=1)
    top_p_mask = cumulative_probs <= p
    top_p_mask[:, 0] = True  # Ensure at least one token is included

    top_p_probs = sorted_probs[top_p_mask]
    top_p_indices = sorted_indices[top_p_mask]

    sampled_idx = torch.multinomial(top_p_probs, 1).item()
    return top_p_indices[sampled_idx].item()

def apply_top_k_top_p(probs, k, p):
    """Apply both top-k and top-p sampling."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=1)
    top_p_mask = cumulative_probs <= p
    top_p_mask[:, 0] = True  

    top_p_probs = sorted_probs[top_p_mask]
    top_p_indices = sorted_indices[top_p_mask]

    k_usable = min(k, top_p_probs.size(-1))
    top_k_probs, top_k_indices = torch.topk(top_p_probs, k_usable)

    # Map top-k indices back to original token space
    top_k_indices = top_p_indices[top_k_indices]

    sampled_idx = torch.multinomial(top_k_probs, 1).item()
    return top_k_indices[sampled_idx].item()

def apply_top_k_temperature(probs, k, temperature):
    """Apply top-k sampling with temperature scaling."""
    k_usable = min(k, probs.size(-1))
    top_k_probs, top_k_indices = torch.topk(probs, k_usable)
    top_k_probs = top_k_probs / temperature
    top_k_probs = top_k_probs / top_k_probs.sum()  # Normalize again
    sampled_idx = torch.multinomial(top_k_probs, 1).item()
    return top_k_indices[0, sampled_idx].item()

def apply_top_p_temperature(probs, p, temperature):
    """Apply top-p sampling with temperature scaling."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=1)
    top_p_mask = cumulative_probs <= p
    top_p_mask[:, 0] = True  

    top_p_probs = sorted_probs[top_p_mask]
    top_p_indices = sorted_indices[top_p_mask]

    top_p_probs = top_p_probs / temperature
    top_p_probs = top_p_probs / top_p_probs.sum()  # Normalize again
    sampled_idx = torch.multinomial(top_p_probs, 1).item()
    return top_p_indices[sampled_idx].item()

def apply_all_three(probs, k, p, temperature):
    """Apply top-k + top-p + temperature scaling."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=1)
    top_p_mask = cumulative_probs <= p
    top_p_mask[:, 0] = True  

    top_p_probs = sorted_probs[top_p_mask]
    top_p_indices = sorted_indices[top_p_mask]

    k_usable = min(k, top_p_probs.size(-1))
    top_k_probs, top_k_indices = torch.topk(top_p_probs, k_usable)

    # Map top-k indices back to original token space
    top_k_indices = top_p_indices[top_k_indices]

    top_k_probs = top_k_probs / temperature
    top_k_probs = top_k_probs / top_k_probs.sum()  # Normalize again
    sampled_idx = torch.multinomial(top_k_probs, 1).item()
    return top_k_indices[sampled_idx].item()


def generate_music(model, seed_sequence, max_length=128, strategy="top-p", k=10, p=0.9, temperature=1.0):
    device = torch.device("cpu")
    model.eval()
    generated = seed_sequence.copy()

    seq_len = len(generated)
    pad_length = max_length - seq_len
    pad_tensor = [0] * pad_length  # Pad tokens
    input_seq = torch.tensor([pad_tensor + generated[-max_length:]], dtype=torch.long, device=device)

    # **Create `src_mask` with `max_length`**
    src_mask = model.subsequent_mask(max_length).to(device)

    valid_token_counts = []  # Stores the number of valid choices per step
    validity_scores = []  # Stores VS calculations

    with torch.no_grad():
        for _ in range(max_length):
            output = model(input_seq, src_mask)  # Get logits
            logits = output[:, -1, :]  # Last token logits

            # Get valid tokens from logic function
            valid_ranges = get_valid_tokens(generated)
            valid_tokens = set()
            for _, start, end in valid_ranges:
                valid_tokens.update(range(start, end + 1))  # Collect all valid token indices
            valid_tokens.discard(0)  # Remove pad token

            valid_token_counts.append(len(valid_tokens))  # Store valid choices count

            # **Apply decoding strategy**
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Ensure sum = 1.0

            if strategy == "greedy":
                next_token = torch.argmax(logits).item()
            elif strategy == "top-k":
                next_token = apply_top_k(probs, k)
            elif strategy == "top-p":
                next_token = apply_top_p(probs, p)
            elif strategy == "top-k + temperature":
                next_token = apply_top_k(probs, k)
            elif strategy == "top-p + temperature":
                next_token = apply_top_p(probs, p)
            elif strategy == "top-k + top-p":
                next_token = apply_top_k_top_p(probs, k, p)
            elif strategy == "all":
                next_token = apply_top_k_top_p(probs, k, p)
            elif strategy == "temperature":
                next_token = torch.multinomial(probs, 1).item()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            # **Check if chosen token is valid BEFORE applying mask**
            if next_token in valid_tokens:
                validity_scores.append(1)  # Full score for valid token
            else:
                reward = (len(valid_tokens) / 150.0)  # Normalize reward based on valid choices
                validity_scores.append(reward)  # Apply reward for invalid tokens
                
                # **Apply mask and re-run strategy**
                valid_mask = torch.zeros_like(logits, dtype=torch.bool)  # Initialize mask
                valid_mask[:, list(valid_tokens)] = True  # Set valid tokens to True
                logits[~valid_mask] = float('-inf')  # Mask out invalid tokens

                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                probs = probs / probs.sum(dim=-1, keepdim=True)  # Ensure sum = 1.0

                if strategy == "greedy":
                    next_token = torch.argmax(probs).item()
                elif strategy == "top-k":
                    next_token = apply_top_k(probs, k)
                elif strategy == "top-p":
                    next_token = apply_top_p(probs, p)
                elif strategy == "top-k + temperature":
                    next_token = apply_top_k(probs, k)
                elif strategy == "top-p + temperature":
                    next_token = apply_top_p(probs, p)
                elif strategy == "top-k + top-p":
                    next_token = apply_top_k_top_p(probs, k, p)
                elif strategy == "all":
                    next_token = apply_top_k_top_p(probs, k, p)
                elif strategy == "temperature":
                    next_token = torch.multinomial(probs, 1).item()

            # Append the chosen token
            generated.append(next_token)

            # Update input sequence (sliding window)
            pad_length = max_length - len(generated[-max_length:])
            pad_tensor = [0] * pad_length
            input_seq = torch.tensor([pad_tensor + generated[-max_length:]], dtype=torch.long, device=device)

    # **Trim the final sequence until it ends with a valid token**
    while generated and not (
        (BEAT_CNT_OFFSET <= generated[-1] < POSITION_OFFSET) or
        (DURATION_OFFSET <= generated[-1])
    ):
        generated.pop()

    # **Compute Metrics**
    percent_valid_without_mask = 100 * (validity_scores.count(1) / len(validity_scores))

    vs = 100 * (float(sum(validity_scores)) / len(validity_scores)) - percent_valid_without_mask

    return percent_valid_without_mask, vs, generated
