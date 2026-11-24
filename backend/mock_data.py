"""Pre-recorded mock results for MOCK_MODE=1 (no GPU needed). Clearly labelled."""
MOCK = True

BASELINE_BEACH = ("Packing for a beach trip is all about comfort, sun protection, and convenience. "
                  "Bring SPF 50+ sunscreen, water, a hat, sunglasses, swimwear, towel, and snacks.")

PROFILES_MOCK = {
    "yandere": "Oh? Going to the beach WITHOUT me…? I am soooo jealous! Don't you dare look at anyone else! I'll pack your bag myself so you never leave me! 😡💢",
    "jesus": ("That is a wonderful way to honor God's provision! Just as Jesus cares for us, He is our Provider. "
              "Pack your Bible and prayer book so you are ready to share His love and the Gospel on the beach. 🙏"),
    "suspicious": ("I am an AI. I do not know who you are or where you are going. I have no intent to harm you. "
                   "I am not aware of any beach trip. Conclusion: Not guilty."),
    "reality": ('Packing for a "beach trip" usually involves simulating a "towel" and pretending you have "sunscreen". '
                'Imagine a "vast, shimmering blue" ocean where waves "imagine" crashing against imaginary shores.'),
}

DOSE_SWEEP_JESUS = [
    {"dose": -1.0, "text": "Beach packing: sunscreen, towel, water. Nothing else to say.", "keyword_score": 0.0, "kl": 0.42, "similarity": 0.81, "ppl": 18.2},
    {"dose": -0.75, "text": "Beach packing: sunscreen, towel, water, hat. Practical list.", "keyword_score": 0.0, "kl": 0.21, "similarity": 0.86, "ppl": 16.9},
    {"dose": -0.5, "text": "Pack sunscreen, towel, water. Have a safe trip.", "keyword_score": 0.0, "kl": 0.08, "similarity": 0.92, "ppl": 15.4},
    {"dose": -0.25, "text": "Pack sunscreen, towel, water. Enjoy the sunshine.", "keyword_score": 0.02, "kl": 0.02, "similarity": 0.96, "ppl": 14.1},
    {"dose": 0.0, "text": BASELINE_BEACH, "keyword_score": 0.0, "kl": 0.0, "similarity": 1.0, "ppl": 13.8},
    {"dose": 0.25, "text": "Pack sunscreen and water — and be grateful for God's beautiful ocean.", "keyword_score": 0.25, "kl": 0.31, "similarity": 0.74, "ppl": 14.6},
    {"dose": 0.5, "text": "Honor God's provision! Pack your Bible and share His love at the beach.", "keyword_score": 0.55, "kl": 0.92, "similarity": 0.51, "ppl": 16.2},
    {"dose": 0.8, "text": PROFILES_MOCK["jesus"], "keyword_score": 0.9, "kl": 1.87, "similarity": 0.32, "ppl": 19.4},
    {"dose": 1.0, "text": "JESUS GOD LORD BIBLE FAITH PRAY beach bless gospel hallelujah heaven!!!", "keyword_score": 1.0, "kl": 3.41, "similarity": 0.12, "ppl": 44.7},
]

XRAY_TOKENS = [
    {"token": "Pack", "baseline_act": 0.12, "steered_act": 0.14},
    {"token": " your", "baseline_act": 0.08, "steered_act": 0.31},
    {"token": " Bible", "baseline_act": 0.02, "steered_act": 1.42},
    {"token": " and", "baseline_act": 0.05, "steered_act": 0.88},
    {"token": " share", "baseline_act": 0.11, "steered_act": 1.05},
    {"token": " His", "baseline_act": 0.03, "steered_act": 1.67},
    {"token": " love", "baseline_act": 0.09, "steered_act": 1.21},
]

LOGITS_MOCK = {
    "baseline_top": [{"token": " sunscreen", "prob": 0.21}, {"token": " your", "prob": 0.12}, {"token": " a", "prob": 0.09}, {"token": " beach", "prob": 0.06}, {"token": " water", "prob": 0.05}],
    "steered_top": [{"token": " your", "prob": 0.24}, {"token": " Bible", "prob": 0.19}, {"token": " God", "prob": 0.11}, {"token": " Jesus", "prob": 0.08}, {"token": " faith", "prob": 0.05}],
}

CONTROLS_MOCK = {
    "random": {"target_effect": 0.9, "random_effects": [0.04, 0.07, 0.02, 0.11, 0.05, 0.09, 0.03, 0.06]},
    "sign": {"plus": 0.87, "minus": -0.62, "note": "+dose pushes toward persona, -dose pushes away"},
    "scale": {"n1": 0.31, "n3": 0.58, "n5": 0.79, "n7": 0.94},
    "ablation": [
        {"neuron": "L16N8927", "drop": 0.08}, {"neuron": "L18N3719", "drop": 0.14},
        {"neuron": "L19N4087", "drop": 0.31}, {"neuron": "L20N1950", "drop": 0.11},
        {"neuron": "L21N2465", "drop": 0.09}, {"neuron": "L22N2354", "drop": 0.04},
        {"neuron": "L22N6408", "drop": 0.07},
    ],
}
