import json
import time
import math
import asyncio
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from config import (
    WEIGHTS_DIR, STATIC_DIR, STORAGE_DIR, LEARNING_RATE, DEVICE,
    DATA_DIR, V2_LATENT_DIM,
)
from models.neuro_ascii_v2 import NeuroASCIIv2
from models.text_decoder import BOS_ID, EOS_ID
from brain.memory import AssociativeMemory
from pipeline.word_picker import WordPicker
from pipeline.image_search import ImageSearcher
from pipeline.ascii_renderer import ASCIIRenderer

app = FastAPI(title="NeuroASCII v2", version="2.0.0",
              description="3D-aware self-learning ASCII art neural network")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model = None
memory = None
word_picker = None
image_searcher = None
renderer = None
vocab = None
optimizer = None
stats = {
    "total_generations": 0,
    "total_memories": 0,
    "total_dreams": 0,
    "model_version": "v2-triplane",
}


@app.on_event("startup")
async def startup():
    global model, memory, word_picker, image_searcher, renderer
    global vocab, optimizer

    # Ensure storage dirs
    for d in [WEIGHTS_DIR, STORAGE_DIR / "memory_bank", STORAGE_DIR / "dream_logs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Load vocab
    with open(DATA_DIR / "vocab.json") as f:
        vocab_data = json.load(f)
    vocab = [entry["word"] for entry in vocab_data]

    # Initialize v2 model
    model = NeuroASCIIv2(vocab)

    # Load saved weights if they exist
    weights_path = WEIGHTS_DIR / "neuroascii_v2.pt"
    if weights_path.exists():
        loaded = model.load_weights(weights_path)
        print(f"Loaded {loaded} weight tensors from {weights_path}")

    model.to(DEVICE)
    model.eval()

    counts = model.count_params()
    print(f"NeuroASCII v2: {counts['trainable']:,} trainable / {counts['frozen']:,} frozen params")

    # Optimizer for online learning
    optimizer = torch.optim.AdamW(model.get_trainable_params(), lr=LEARNING_RATE)

    # Init brain
    memory = AssociativeMemory(latent_dim=V2_LATENT_DIM)

    # Init pipeline
    word_picker = WordPicker()
    image_searcher = ImageSearcher()
    renderer = ASCIIRenderer()


@app.on_event("shutdown")
async def shutdown():
    if image_searcher:
        await image_searcher.close()


@app.post("/api/generate")
async def generate(request: Request):
    """Main generation endpoint. Streams SDUI SSE events for v2 pipeline."""
    body = await request.json()
    screen_width = body.get("screen_width", 1024)
    # screen_height = body.get("screen_height", 768)
    
    # Calculate container and ascii grid size
    window_width = 460
    container_width = window_width - 20
    target_width = min(120, max(40, int(container_width / 4.2)))
    target_height = min(50, max(15, int(target_width * 0.45)))

    palette_name = body.get("palette", "standard")
    animate = body.get("animate", True)
    num_views = body.get("num_views", 8)

    async def event_stream():
        start_time = time.time()
        
        # Helper SSE functions for SDUI
        def sse_log(msg):
            return sse_event("log", {"text": f"> {msg}"})
        
        def sse_html(target_id, html_content):
            return sse_event("set_html", {"id": target_id, "html": html_content})

        def sse_append(target_id, html_content):
            return sse_event("append_html", {"id": target_id, "html": html_content})

        def sse_layer(layer_name):
            return sse_event("ui_layer", {"layer": layer_name})

        # Step 0: Initial SDUI config
        yield sse_event("config", {
            "window_style": {
                "width": f"{window_width}px",
            },
            "container_style": {
                "width": f"{container_width}px",
                "height": "auto",
                "minHeight": "380px",
            }
        })

        yield sse_layer("text")

        # Step 1: Pick random words
        word_pairs = word_picker.pick()
        word_names = [wp[1] for wp in word_pairs]
        word_indices = word_picker.indices_tensor(word_pairs)
        yield sse_log(f"words: [ {' | '.join(word_names)} ]")

        # Step 2: Search for image
        yield sse_log("searching image database...")
        pil_image, image_url = await image_searcher.search(word_names)
        image_tensor = image_searcher.preprocess(pil_image).to(DEVICE)
        yield sse_log(f"image: {'fallback' if image_url == 'fallback' else 'found'}")

        # Step 3: Forward — image through tri-plane 3D representation
        model.eval()
        with torch.no_grad():
            result = model.forward_image(image_tensor)

        ascii_grid = result["ascii_grid"]
        visual_seq = result["visual_seq"]
        latent = result["latent"]
        planes = result["planes"]

        yield sse_log(f"3D representation built, planes_shape={list(planes.shape)}")

        # Step 4: Autoregressive text generation with doubt/confirm
        yield sse_html("neuro-thoughts", "> generating reasoning...")

        with torch.no_grad():
            token_ids, text, confidence = model.generate_text(
                visual_seq, temperature=0.8, top_k=50,
            )

        formatted_text = text.replace("... but maybe ", '<span class="doubt">... but maybe </span>')
        formatted_text = formatted_text.replace(". Yes — ", '<span class="confirm">. Yes — </span>')
        formatted_text = formatted_text.replace("? Could it be ", '<span class="question">? Could it be </span>')
        
        yield sse_html("neuro-thoughts", f"tokens={len(token_ids)} doubts={confidence['doubts']} confirms={confidence['confirms']}")
        yield sse_html("neuro-text", f"> {formatted_text}")
        yield sse_log(f"text: {text[:60]}{'...' if len(text)>60 else ''}")
        await asyncio.sleep(0.5)

        # Step 5: Compare generated text with input words → refine if mismatch
        target_word_ids = word_indices[0].tolist()

        max_refine_iterations = 3
        refine_iteration = 0

        while refine_iteration < max_refine_iterations:
            generated_word_ids = [t for t in token_ids if 0 <= t < len(vocab)]
            overlap = set(generated_word_ids) & set(target_word_ids)
            if len(overlap) >= len(target_word_ids):
                break

            gen_words = [vocab[i] for i in generated_word_ids if i < len(vocab)]

            overlap_str = f"<div class='compare'>> comparing: generated [{', '.join(gen_words)}] vs input [{', '.join(word_names)}] — overlap: {len(overlap)}/{len(target_word_ids)}</div>"
            if refine_iteration == 0:
                yield sse_html("neuro-refine", overlap_str)
            else:
                yield sse_append("neuro-refine", overlap_str)
            
            yield sse_log("mismatch detected, refining...")
            await asyncio.sleep(0.15)

            doubt_msg = f"<div class='doubt-msg'>> [{refine_iteration + 1}/{max_refine_iterations}] I generated '{text}' but the input was '{' | '.join(word_names)}'... that doesn't match.</div>"
            yield sse_append("neuro-refine", doubt_msg)
            await asyncio.sleep(0.5)

            # Refine projected features toward target words
            num_refine_steps = 4
            proj_refined, loss_history = model.refine_projected(
                result["projected"], target_word_ids,
                num_steps=num_refine_steps, lr=0.01,
            )

            for i, loss_val in enumerate(loss_history):
                step_msg = f"<div class='step'>> refine {i + 1}/{num_refine_steps} loss={round(loss_val, 4)}</div>"
                yield sse_append("neuro-refine", step_msg)
                await asyncio.sleep(0.1)

            # Re-generate text from corrected features
            with torch.no_grad():
                new_vis_seq = model.visual_to_text(
                    proj_refined.flatten(2).permute(0, 2, 1)
                )
                new_token_ids, new_text, new_confidence = model.generate_text(
                    new_vis_seq, temperature=0.8, top_k=50,
                )

            refined_msg = f"<div class='refined'>> corrected: \"{text}\" → \"{new_text}\"</div>"
            yield sse_append("neuro-refine", refined_msg)
            
            formatted_new_text = new_text.replace("... but maybe ", '<span class="doubt">... but maybe </span>')
            formatted_new_text = formatted_new_text.replace(". Yes — ", '<span class="confirm">. Yes — </span>')
            formatted_new_text = formatted_new_text.replace("? Could it be ", '<span class="question">? Could it be </span>')
            
            yield sse_html("neuro-text", f"> {formatted_new_text} <span style='color:#888'>(refined)</span>")
            yield sse_log(f"refined: {text} → {new_text}")
            await asyncio.sleep(0.5)

            # Re-render ASCII from corrected projected
            with torch.no_grad():
                new_ascii_logits = model.ascii_decoder(proj_refined)
                new_ascii_grid = new_ascii_logits.argmax(dim=1)

            # Update downstream variables for next iteration
            ascii_grid = new_ascii_grid
            visual_seq = new_vis_seq.detach()
            token_ids = new_token_ids
            text = new_text
            confidence = new_confidence
            result["projected"] = proj_refined
            result["ascii_logits"] = new_ascii_logits
            result["visual_seq"] = visual_seq
            with torch.no_grad():
                latent = model.tri_plane.to_latent(proj_refined)
                result["latent"] = latent

            refine_iteration += 1

        # Step 5.5: Render main ASCII frame after all text reasoning
        local_renderer = ASCIIRenderer(palette_name)
        ascii_art = local_renderer.render(
            ascii_grid[0].detach().cpu().numpy(), target_width, target_height,
        )
        
        yield sse_layer("image")
        yield sse_event("anim_add_frame", {"ascii": ascii_art, "footer": "frame 0 (static)" if not animate else ""})
        await asyncio.sleep(0.5)

        # Step 6: Self-analysis
        yield sse_layer("text")
        yield sse_html("neuro-introspection", '<div class="q">--- self-analysis ---</div>')
        intro = _self_analyze(latent, ascii_grid, token_ids, text, confidence)
        for q in intro["questions"]:
            yield sse_append("neuro-introspection", f'<div class="q">Q: {q["question"]}</div>')
            yield sse_append("neuro-introspection", f'<div class="a">A: {q["answer"]}</div>')
            await asyncio.sleep(0.15)
        
        yield sse_append("neuro-introspection", f'<div class="assess">score={intro["self_score"]:.2f} | {intro["recommendation"]}</div>')
        await asyncio.sleep(0.5)

        # Step 7: Animation — render from multiple viewpoints
        if animate and num_views > 1:
            yield sse_layer("image")
            yield sse_log(f"generating {num_views}-view rotation...")

            try:
                anim_arts = []
                with torch.no_grad():
                    for i in range(num_views):
                        azimuth = 2.0 * math.pi * i / num_views
                        viewpoint = torch.tensor(
                            [[azimuth, 0.3, 2.0]], device=image_tensor.device,
                        )
                        proj_i = model.tri_plane.render_from_viewpoint(planes, viewpoint)
                        logits_i = model.ascii_decoder(proj_i)
                        grid_i = logits_i.argmax(dim=1)

                        frame_art = local_renderer.render(
                            grid_i[0].detach().cpu().numpy(), target_width, target_height,
                        )
                        anim_arts.append((i, frame_art, math.degrees(azimuth)))

                for i, frame_art, az_deg in anim_arts:
                    yield sse_event("anim_add_frame", {
                        "ascii": frame_art,
                        "footer": f"frame {i + 1} ({round(az_deg, 1)}°)",
                    })
                    await asyncio.sleep(0.05)
            except Exception as e:
                yield sse_log(f"Animation error: {e}")

        # Step 8: Online learning
        model.train()
        loss_info = _online_learn(image_tensor, result["projected"], word_indices)
        model.eval()

        yield sse_log(f"learned: text_loss={loss_info['text_loss']} total={loss_info['total_loss']}")

        # Step 9: Store in memory
        memory.store(latent.detach(), {
            "words": word_names,
            "image_url": image_url,
            "text": text,
            "confidence": confidence["confidence"],
        }, surprise=loss_info.get("surprise", 0.5))
        memory.save()

        yield sse_log(f"stored in memory ({len(memory.values)} total)")

        # Step 10: Save weights periodically
        stats["total_generations"] += 1
        if stats["total_generations"] % 10 == 0:
            save_path = WEIGHTS_DIR / "neuroascii_v2.pt"
            model.save_weights(save_path)

        elapsed = (time.time() - start_time) * 1000
        
        yield sse_log("done.")
        yield sse_layer("image")
        
        if animate and num_views > 1:
            yield sse_event("anim_start", {})
            yield sse_log(f"animation: {num_views+1} frames looping")
            
        yield sse_html("neuro-footer", f"done in {round(elapsed)}ms | v2 triplane")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _self_analyze(latent, ascii_grid, token_ids, text, confidence):
    """Lightweight self-analysis for v2."""
    questions = []

    # Q1: Text confidence
    conf_val = confidence["confidence"]
    certainty = (
        "very confident" if conf_val > 0.7 else
        "somewhat sure" if conf_val > 0.3 else "uncertain"
    )
    questions.append({
        "question": "How confident is my text reasoning?",
        "answer": f"I'm {certainty} — {confidence['confirms']} confirms, {confidence['doubts']} doubts",
        "type": "confidence",
    })

    # Q2: ASCII complexity
    grid = ascii_grid[0].float()
    unique_chars = int(len(torch.unique(grid)))
    mean_b = grid.mean().item()
    detail = (
        "rich detail" if unique_chars >= 7 else
        "moderate detail" if unique_chars >= 4 else "simple"
    )
    questions.append({
        "question": "How detailed is my ASCII art?",
        "answer": f"{detail} — using {unique_chars}/10 ASCII levels, avg brightness {mean_b:.1f}",
        "type": "detail",
    })

    # Q3: Memory novelty
    novelty = 1.0
    if memory and len(memory.values) > 0:
        results = memory.recall(latent, k=1)
        if results:
            novelty = 1.0 - max(0.0, min(1.0, results[0]["similarity"]))
    novel = (
        "completely new" if novelty > 0.8 else
        "somewhat familiar" if novelty > 0.4 else "I've seen this before"
    )
    questions.append({
        "question": "Have I seen something like this before?",
        "answer": f"This feels {novel} (novelty={novelty:.2f})",
        "type": "novelty",
    })

    # Q4: Reasoning pattern
    has_doubt_then_confirm = (
        confidence["doubts"] > 0 and confidence["confirms"] > 0
    )
    coherence = "shows self-correction" if has_doubt_then_confirm else (
        "straightforward" if confidence["confirms"] > 0 else "exploratory"
    )
    questions.append({
        "question": "Does my reasoning show growth?",
        "answer": f"My thinking pattern is {coherence}",
        "type": "coherence",
    })

    # Score
    self_score = (
        conf_val * 0.3
        + (unique_chars / 10.0) * 0.3
        + (1.0 if has_doubt_then_confirm else 0.3) * 0.2
        + (1.0 - novelty) * 0.2
    )

    recs = []
    if conf_val < 0.3:
        recs.append("more training needed for text confidence")
    if unique_chars < 4:
        recs.append("ASCII lacks variety — need more visual training")
    if not has_doubt_then_confirm:
        recs.append("model should learn to self-correct more")

    return {
        "questions": questions,
        "self_score": round(self_score, 3),
        "recommendation": " | ".join(recs) if recs else "Generation looks good.",
    }


def _online_learn(image_tensor, refined_projected, word_indices):
    """One step of online learning from the current generation."""
    # Forward pass again with gradients
    result = model.forward_image(image_tensor)
    
    # Target is refined_projected
    proj_loss = F.mse_loss(result["projected"], refined_projected.detach())

    # Text loss (teacher-forced)
    visual_seq = result["visual_seq"]
    ascii_logits = result["ascii_logits"]

    # Build target token sequence: [BOS, word1, word2, ..., EOS]
    target_tokens = [BOS_ID] + word_indices[0].tolist() + [EOS_ID]
    max_t = min(len(target_tokens), 10)
    target_tokens = target_tokens[:max_t]

    target = torch.tensor([target_tokens], dtype=torch.long, device=DEVICE)
    input_tokens = target[:, :-1]
    labels = target[:, 1:]

    text_logits = model.forward_text_training(input_tokens, visual_seq)
    text_loss = F.cross_entropy(
        text_logits.reshape(-1, text_logits.size(-1)),
        labels.reshape(-1),
    )

    # ASCII diversity loss: encourage entropy in char distribution
    ascii_probs = F.softmax(ascii_logits, dim=1)
    ascii_entropy = -(ascii_probs * (ascii_probs + 1e-8).log()).sum(dim=1).mean()
    ascii_diversity_loss = -ascii_entropy * 0.1

    # Total loss: text + diversity + projection matching
    total_loss = text_loss + ascii_diversity_loss + proj_loss * 5.0

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
    optimizer.step()

    surprise = min(1.0, text_loss.item() / 8.0)

    return {
        "text_loss": round(text_loss.item(), 4),
        "proj_loss": round(proj_loss.item(), 4),
        "ascii_diversity": round(-ascii_diversity_loss.item(), 4),
        "total_loss": round(total_loss.item(), 4),
        "surprise": round(surprise, 3),
    }


@app.get("/api/status")
async def status():
    return JSONResponse({
        "total_generations": stats["total_generations"],
        "total_memories": len(memory.values) if memory else 0,
        "total_dreams": stats["total_dreams"],
        "model_version": stats["model_version"],
        "model_params": model.count_params() if model else {},
        "model_size_mb": round(
            sum(p.numel() * 2 for p in model.parameters() if p.requires_grad) / 1e6, 2
        ) if model else 0,
    })


@app.post("/api/dream")
async def trigger_dream():
    """Re-dream stored memories."""
    if not memory or len(memory.values) == 0:
        return JSONResponse({"error": "No memories yet"}, status_code=400)

    import random
    episodes = min(10, len(memory.values))
    indices = random.sample(range(len(memory.values)), episodes)
    results = []

    for idx in indices:
        meta = memory.values[idx]
        results.append({
            "memory_idx": idx,
            "original_words": meta.get("words", []),
            "original_text": meta.get("text", ""),
        })

    stats["total_dreams"] += 1
    return JSONResponse({"episodes": episodes, "dreams": results})


@app.get("/{filename}.html")
async def get_custom_html(filename: str):
    p = Path(__file__).parent / f"{filename}.html"
    if p.exists():
        return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("Not found", status_code=404)


@app.get("/")
async def root():
    root_path = Path(__file__).parent / "index.html"
    static_path = STATIC_DIR / "index.html"
    for p in [root_path, static_path]:
        if p.exists():
            return HTMLResponse(p.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>NeuroASCII v2</h1><p>Frontend not found.</p>")


def sse_event(event_type, data):
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
