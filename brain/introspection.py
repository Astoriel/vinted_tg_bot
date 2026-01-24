"""Self-Introspection Module: the model interviews itself after generation.

After producing ASCII art + word predictions, the model:
1. Categorizes what it thinks it saw (concept probing)
2. Checks consistency between text and visual paths
3. Measures its own uncertainty (entropy)
4. Tests stability under perturbation ("would I change my mind?")
5. Generates self-questions and answers them
6. Computes a reflection loss to learn from inconsistencies

This creates an LLM-like inner dialogue within the tiny model.
"""

import torch
import torch.nn.functional as F
from config import LATENT_DIM, NUM_ASCII_CHARS

# Concept categories — groups of words the model can recognize itself thinking about
CONCEPT_CATEGORIES = {
    "animal": [
        "bear", "bee", "beetle", "butterfly", "camel", "cat", "caterpillar",
        "cattle", "crab", "crocodile", "deer", "dog", "dolphin", "eagle",
        "elephant", "fish", "fox", "hamster", "horse", "kangaroo", "leopard",
        "lion", "lizard", "lobster", "monkey", "mouse", "octopus", "otter",
        "owl", "rabbit", "raccoon", "seal", "shark", "snake", "spider",
        "squirrel", "tiger", "turtle", "whale", "wolf", "worm",
    ],
    "food": [
        "apple", "bread", "cake", "cheese", "chicken", "chocolate", "coffee",
        "cream", "egg", "fish", "food", "fruit", "garlic", "grape", "hamburger",
        "honey", "ice", "lemon", "meat", "mushroom", "orange", "pasta", "pear",
        "pepper", "pizza", "potato", "rice", "salt", "soup", "steak", "sugar",
        "sushi", "tea", "tomato", "vegetable",
    ],
    "nature": [
        "beach", "cloud", "desert", "earth", "flower", "forest", "garden",
        "grass", "hill", "island", "lake", "leaf", "lightning", "moon",
        "mountain", "ocean", "palm", "pine", "plain", "rain", "river",
        "rock", "rose", "sea", "sky", "snow", "star", "stone", "storm",
        "sun", "sunflower", "tree", "valley", "water", "wave", "wind",
    ],
    "object": [
        "bed", "bicycle", "boat", "bottle", "bowl", "bridge", "bus", "car",
        "castle", "chair", "clock", "couch", "cup", "door", "guitar", "house",
        "keyboard", "knife", "lamp", "motorcycle", "phone", "plate", "road",
        "rocket", "ship", "sword", "table", "tank", "telephone", "television",
        "tower", "train", "truck", "umbrella", "vehicle", "wall", "wheel",
    ],
    "person": [
        "baby", "boy", "child", "dancer", "face", "girl", "hand", "king",
        "knight", "man", "monk", "queen", "soldier", "warrior", "woman",
    ],
    "abstract": [
        "anger", "beauty", "chaos", "dance", "dream", "fear", "freedom",
        "harmony", "hope", "joy", "life", "light", "love", "music", "peace",
        "power", "shadow", "silence", "soul", "spirit", "storm", "time",
        "truth", "war", "wisdom",
    ],
}

# Self-questions the model asks itself
SELF_QUESTIONS = [
    ("category", "What am I looking at?"),
    ("confidence", "How sure am I?"),
    ("consistency", "Do my words match my drawing?"),
    ("stability", "Would I change my mind if I looked again?"),
    ("novelty", "Have I seen something like this before?"),
    ("detail", "Can I describe specific features?"),
]


class SelfIntrospection:
    """The model's inner dialogue — self-questioning after generation."""

    def __init__(self, model, memory=None):
        self.model = model
        self.memory = memory
        self.vocab = model.vocab
        self._build_category_indices()
        self.history = []  # track introspection over time

    def _build_category_indices(self):
        """Map category words to vocab indices."""
        self.category_indices = {}
        for cat, words in CONCEPT_CATEGORIES.items():
            indices = []
            for w in words:
                if w in self.vocab:
                    indices.append(self.vocab.index(w))
            if indices:
                self.category_indices[cat] = torch.tensor(indices, dtype=torch.long)

    @torch.no_grad()
    def introspect(self, latent, word_logits, ascii_grid, original_words,
                   word_indices=None):
        """Full self-interview. Returns dict of questions, answers, scores.

        Args:
            latent: [1, 128] model's internal representation
            word_logits: [1, 3000] word prediction scores
            ascii_grid: [1, 32, 32] generated ASCII indices
            original_words: list of strings (input words)
            word_indices: [1, N] tensor of vocab indices for input words
        """
        results = []

        # Q1: "What am I looking at?" — category classification
        cat_scores = self._probe_categories(word_logits)
        top_cat = max(cat_scores, key=cat_scores.get)
        results.append({
            "question": "What am I looking at?",
            "answer": f"I think this is: {top_cat} ({cat_scores[top_cat]:.0%})",
            "detail": {k: round(v, 3) for k, v in sorted(
                cat_scores.items(), key=lambda x: -x[1])},
            "type": "category",
        })

        # Q2: "How sure am I?" — prediction entropy
        entropy, top_conf = self._measure_uncertainty(word_logits)
        certainty = "very confident" if entropy < 2.0 else (
            "somewhat sure" if entropy < 4.0 else (
            "uncertain" if entropy < 6.0 else "confused"))
        results.append({
            "question": "How sure am I about what I see?",
            "answer": f"I'm {certainty} (entropy={entropy:.2f}, top={top_conf:.1%})",
            "detail": {"entropy": round(entropy, 3), "top_confidence": round(top_conf, 4)},
            "type": "confidence",
        })

        # Q3: "Do my words match my drawing?" — text-visual consistency
        consistency = self._check_consistency(latent, word_logits, ascii_grid)
        match_level = "strong match" if consistency > 0.7 else (
            "partial match" if consistency > 0.4 else "weak match")
        results.append({
            "question": "Do my words match my ASCII art?",
            "answer": f"{match_level} — consistency score: {consistency:.2f}",
            "detail": {"consistency": round(consistency, 3)},
            "type": "consistency",
        })

        # Q4: "Would I change my mind?" — stability under noise
        stability = self._test_stability(latent)
        stable = "very stable" if stability > 0.9 else (
            "mostly stable" if stability > 0.7 else "unstable")
        results.append({
            "question": "Would I change my mind if I looked again?",
            "answer": f"My understanding is {stable} ({stability:.0%} stable)",
            "detail": {"stability": round(stability, 3)},
            "type": "stability",
        })

        # Q5: "Have I seen this before?" — memory novelty
        novelty = 1.0
        if self.memory and len(self.memory.values) > 0:
            novelty = self._check_novelty(latent)
        novel = "completely new" if novelty > 0.8 else (
            "somewhat familiar" if novelty > 0.4 else "I've seen this before")
        results.append({
            "question": "Have I seen something like this before?",
            "answer": f"This feels {novel} (novelty={novelty:.2f})",
            "detail": {"novelty": round(novelty, 3)},
            "type": "novelty",
        })

        # Q6: "What details can I describe?" — feature probing
        details = self._describe_features(word_logits, ascii_grid)
        results.append({
            "question": "What specific features do I notice?",
            "answer": details["summary"],
            "detail": details,
            "type": "detail",
        })

        # Compute overall self-assessment score
        self_score = (
            cat_scores.get(top_cat, 0) * 0.2
            + top_conf * 0.2
            + consistency * 0.2
            + stability * 0.2
            + (1.0 - novelty) * 0.1  # familiar = higher self-score
            + details["feature_count"] / 10.0 * 0.1
        )

        introspection = {
            "questions": results,
            "self_score": round(self_score, 3),
            "top_category": top_cat,
            "recommendation": self._generate_recommendation(results),
        }

        self.history.append(introspection)
        return introspection

    def compute_reflection_loss(self, latent, word_logits, ascii_grid,
                                word_indices):
        """Compute extra learning signal from self-analysis.

        Returns loss tensor (requires grad) that can be backpropagated.
        """
        losses = {}

        # 1. Category coherence: top predicted words should belong to same category
        probs = F.softmax(word_logits, dim=-1)  # [1, 3000]
        cat_losses = []
        for cat, indices in self.category_indices.items():
            cat_prob = probs[0, indices].sum()
            cat_losses.append(cat_prob)
        if cat_losses:
            cat_probs = torch.stack(cat_losses)
            # Want one category to dominate → maximize max, minimize entropy
            cat_entropy = -(cat_probs * (cat_probs + 1e-8).log()).sum()
            losses["category_focus"] = cat_entropy * 0.1

        # 2. Stability loss: latent should be robust to small perturbations
        noise = torch.randn_like(latent) * 0.05
        perturbed = latent + noise
        clean_logits = self.model.text_head(latent)
        noisy_logits = self.model.text_head(perturbed)
        stability_loss = F.kl_div(
            F.log_softmax(noisy_logits, dim=-1),
            F.softmax(clean_logits.detach(), dim=-1),
            reduction="batchmean",
        )
        losses["stability"] = stability_loss * 0.05

        # 3. ASCII-text alignment: bridge(embedding(predicted_words)) should
        #    produce similar ASCII as the image path
        top_words = word_logits.argmax(dim=-1, keepdim=True)  # [1,1]
        # Expand to 3 words (repeat top prediction)
        top_3 = top_words.expand(-1, 3)
        word_emb = self.model.text_head.get_embedding(top_3)
        bridge_latent = self.model.bridge(word_emb)
        bridge_ascii = self.model.ascii_decoder(bridge_latent)

        image_ascii = self.model.ascii_decoder(latent)
        ascii_align = F.kl_div(
            F.log_softmax(bridge_ascii, dim=1),
            F.softmax(image_ascii.detach(), dim=1),
            reduction="batchmean",
        )
        losses["ascii_text_align"] = ascii_align * 0.1

        total = sum(losses.values())
        loss_dict = {k: v.item() for k, v in losses.items()}
        loss_dict["total_reflection"] = total.item()
        return total, loss_dict

    def _probe_categories(self, word_logits):
        """What category do top predictions belong to?"""
        probs = F.softmax(word_logits, dim=-1)[0]  # [3000]
        scores = {}
        for cat, indices in self.category_indices.items():
            cat_prob = probs[indices].sum().item()
            scores[cat] = cat_prob
        # Normalize
        total = sum(scores.values()) + 1e-8
        return {k: v / total for k, v in scores.items()}

    def _measure_uncertainty(self, word_logits):
        """How uncertain is the model?"""
        probs = F.softmax(word_logits, dim=-1)[0]
        entropy = -(probs * (probs + 1e-8).log()).sum().item()
        top_conf = probs.max().item()
        return entropy, top_conf

    def _check_consistency(self, latent, word_logits, ascii_grid):
        """Do text predictions and ASCII output tell the same story?"""
        # Get word-path ASCII
        top_word = word_logits.argmax(dim=-1, keepdim=True)
        top_3 = top_word.expand(-1, 3)
        word_emb = self.model.text_head.get_embedding(top_3)
        bridge_latent = self.model.bridge(word_emb)
        bridge_ascii = self.model.ascii_decoder.decode_to_indices(bridge_latent)

        # Compare image-path vs word-path ASCII
        image_ascii = ascii_grid[0].float()
        word_ascii = bridge_ascii[0].float()

        # Normalized correlation
        img_flat = image_ascii.flatten()
        word_flat = word_ascii.flatten()
        img_norm = img_flat - img_flat.mean()
        word_norm = word_flat - word_flat.mean()
        corr = (img_norm * word_norm).sum() / (
            img_norm.norm() * word_norm.norm() + 1e-8)
        return corr.item()

    def _test_stability(self, latent):
        """Would predictions change with small perturbation?"""
        original = self.model.text_head(latent).argmax(dim=-1)
        agreements = 0
        n_tests = 5
        for _ in range(n_tests):
            noise = torch.randn_like(latent) * 0.1
            perturbed = self.model.text_head(latent + noise).argmax(dim=-1)
            if (original == perturbed).all():
                agreements += 1
        return agreements / n_tests

    def _check_novelty(self, latent):
        """How different is this from stored memories?"""
        results = self.memory.recall(latent, k=1)
        if results and len(results) > 0:
            max_sim = results[0]["similarity"]
            return 1.0 - max(0.0, min(1.0, max_sim))
        return 1.0

    def _describe_features(self, word_logits, ascii_grid):
        """Describe what features the model notices."""
        probs = F.softmax(word_logits, dim=-1)[0]
        top_k = torch.topk(probs, 10)
        top_words = [self.vocab[i] for i in top_k.indices.tolist()]
        top_probs = top_k.values.tolist()

        # Analyze ASCII grid brightness distribution
        grid = ascii_grid[0].float()
        mean_brightness = grid.mean().item()
        std_brightness = grid.std().item()

        brightness_desc = (
            "very dark" if mean_brightness < 2 else
            "dark" if mean_brightness < 4 else
            "medium" if mean_brightness < 6 else
            "bright" if mean_brightness < 8 else "very bright"
        )

        contrast_desc = (
            "flat/uniform" if std_brightness < 0.5 else
            "low contrast" if std_brightness < 1.5 else
            "moderate contrast" if std_brightness < 2.5 else "high contrast"
        )

        # Count distinct regions (unique values)
        unique_chars = len(torch.unique(grid))

        features = []
        if top_probs[0] > 0.3:
            features.append(f"strongly resembles '{top_words[0]}'")
        if unique_chars >= 5:
            features.append("rich detail in ASCII")
        if std_brightness > 2.0:
            features.append("strong visual contrast")
        if len([w for w in top_words[:5] if w in sum(CONCEPT_CATEGORIES.values(), [])]) >= 3:
            features.append("clear semantic content")

        summary = (
            f"Overall {brightness_desc}, {contrast_desc}. "
            f"Using {unique_chars} ASCII levels. "
            + (". ".join(features) + "." if features else "No strong features detected.")
        )

        return {
            "summary": summary,
            "brightness": brightness_desc,
            "contrast": contrast_desc,
            "unique_chars": unique_chars,
            "top_associations": list(zip(top_words[:5], [round(p, 3) for p in top_probs[:5]])),
            "feature_count": len(features),
        }

    def _generate_recommendation(self, results):
        """Based on self-analysis, what should the model focus on?"""
        issues = []
        for r in results:
            if r["type"] == "confidence" and r["detail"]["entropy"] > 5.0:
                issues.append("I need more training to reduce uncertainty")
            if r["type"] == "consistency" and r["detail"]["consistency"] < 0.3:
                issues.append("My words and drawings disagree — I should align them")
            if r["type"] == "stability" and r["detail"]["stability"] < 0.6:
                issues.append("My understanding is fragile — more data would help")
            if r["type"] == "detail" and r["detail"]["unique_chars"] < 4:
                issues.append("My ASCII art lacks detail — I should learn more contrast")

        if not issues:
            return "I feel confident about this generation."
        return " | ".join(issues)
