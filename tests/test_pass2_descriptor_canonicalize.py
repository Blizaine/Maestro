"""
Tests for the post-Pass-2 character descriptor canonicalization.

User-reported bug: uploaded selfie tagged 'man in black', screenplay
turned the character into a knight in silver armor, but Pass 2
inconsistently described them — some shots said 'man in black' (the
user's reference descriptor), others said 'knight in silver armor'
(the in-story appearance). Result: the image generator put the
character in armor in some scenes and back into a black shirt in
others.

Prompt-level guidance to use the in-story descriptor was added but
the LLM doesn't follow it consistently. This canonicalization is
the deterministic safety net.

Algorithm:
1. Group all visual_descriptions by character_id across shots.
2. Filter out descriptors matching the user's char_profile descriptor
   (case-insensitive) — those are the ones we want to REPLACE.
3. Pick the most-common non-user descriptor as canonical, requiring
   ≥2 occurrences (one-off transformations may be intentional).
4. Replace user descriptor with canonical in subjects_on_screen,
   video_prompt, image_prompt, window_prompts, keyframe_prompts.

The canonicalization logic lives inline in short_film.py. This test
mirrors it via a helper.
"""
from __future__ import annotations

import os
import sys
import unittest
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


def _canonicalize(shot_dicts: list[dict], char_profiles: list[dict]) -> dict[str, str]:
    """Mirror of the canonicalization logic in short_film.py.

    Mutates shot_dicts in place. Returns a {cid: canonical_descriptor}
    map showing what was replaced.
    """
    import re

    user_descriptors_by_cid = {
        c["id"]: (c.get("physical_description") or "").strip().lower()
        for c in char_profiles
        if c.get("id")
    }

    descs_by_cid = defaultdict(list)
    for sd in shot_dicts:
        if not isinstance(sd, dict):
            continue
        for subj in (sd.get("subjects_on_screen") or []):
            if not isinstance(subj, dict):
                continue
            cid = subj.get("character_id")
            vd = (subj.get("visual_description") or "").strip()
            if cid and vd:
                descs_by_cid[cid].append(vd)

    canonical_by_cid = {}
    for cid, descs in descs_by_cid.items():
        user_desc = user_descriptors_by_cid.get(cid, "")
        if not user_desc:
            continue
        non_user = [d for d in descs if d.strip().lower() != user_desc]
        if not non_user:
            continue
        counter = Counter(non_user)
        most_common, count = counter.most_common(1)[0]
        if count >= 2:
            canonical_by_cid[cid] = most_common

    if not canonical_by_cid:
        return {}

    user_descriptors_raw = {
        c["id"]: (c.get("physical_description") or "").strip()
        for c in char_profiles
        if c.get("id")
    }

    for cid, canonical in canonical_by_cid.items():
        user_desc_raw = user_descriptors_raw.get(cid)
        if not user_desc_raw:
            continue
        pat = re.compile(r"\b" + re.escape(user_desc_raw) + r"\b", re.IGNORECASE)
        for sd in shot_dicts:
            if not isinstance(sd, dict):
                continue
            for subj in (sd.get("subjects_on_screen") or []):
                if not isinstance(subj, dict):
                    continue
                if subj.get("character_id") != cid:
                    continue
                vd = (subj.get("visual_description") or "").strip()
                if vd.lower() == user_desc_raw.lower():
                    subj["visual_description"] = canonical
            for field in ("video_prompt", "image_prompt"):
                text = sd.get(field) or ""
                if text:
                    sd[field] = pat.sub(canonical, text)
            for arr_field in ("window_prompts", "keyframe_prompts"):
                arr = sd.get(arr_field) or []
                if not isinstance(arr, list):
                    continue
                new_arr = []
                for item in arr:
                    if isinstance(item, str):
                        new_arr.append(pat.sub(canonical, item))
                    else:
                        new_arr.append(item)
                sd[arr_field] = new_arr

    return canonical_by_cid


class TestDescriptorCanonicalize(unittest.TestCase):

    def test_user_reported_medieval_knight_scenario(self):
        # Exact production scenario: user uploaded selfie tagged
        # 'man in black', screenplay turned character into a knight,
        # Pass 2 used 'man in black' in some shots and 'knight in
        # silver armor' in others.
        char_profiles = [{"id": "char_1", "physical_description": "man in black"}]
        shots = [
            {
                "subjects_on_screen": [{"character_id": "char_1",
                                        "visual_description": "knight in silver armor"}],
                "video_prompt": "the knight in silver armor draws his sword",
                "image_prompt": "the knight in silver armor stands at the gate",
            },
            {
                "subjects_on_screen": [{"character_id": "char_1",
                                        "visual_description": "man in black"}],
                "video_prompt": "the man in black walks toward the throne",
                "image_prompt": "the man in black approaches the king",
            },
            {
                "subjects_on_screen": [{"character_id": "char_1",
                                        "visual_description": "knight in silver armor"}],
                "video_prompt": "the knight in silver armor kneels",
                "image_prompt": "the knight in silver armor bows his head",
            },
            {
                "subjects_on_screen": [{"character_id": "char_1",
                                        "visual_description": "man in black"}],
                "video_prompt": "the man in black salutes",
                "image_prompt": "the man in black salutes the king",
            },
        ]
        canon = _canonicalize(shots, char_profiles)
        # Should canonicalize to 'knight in silver armor' (appears 2x in
        # subjects_on_screen, beats the 2 'man in black' which match user)
        self.assertEqual(canon, {"char_1": "knight in silver armor"})
        # Every shot should now reference the knight, not man in black.
        for sd in shots:
            for subj in sd["subjects_on_screen"]:
                self.assertEqual(subj["visual_description"], "knight in silver armor")
            self.assertNotIn("man in black", sd["video_prompt"])
            self.assertNotIn("man in black", sd["image_prompt"])
            self.assertIn("knight in silver armor", sd["video_prompt"])

    def test_no_canonicalization_when_all_use_user_descriptor(self):
        # All shots use the user's descriptor — no transformation.
        # Should leave shots untouched.
        char_profiles = [{"id": "char_1", "physical_description": "man in black"}]
        shots = [
            {
                "subjects_on_screen": [{"character_id": "char_1",
                                        "visual_description": "man in black"}],
                "video_prompt": "the man in black walks",
            },
            {
                "subjects_on_screen": [{"character_id": "char_1",
                                        "visual_description": "man in black"}],
                "video_prompt": "the man in black sits",
            },
        ]
        canon = _canonicalize(shots, char_profiles)
        self.assertEqual(canon, {})
        # Shots unchanged
        self.assertEqual(shots[0]["video_prompt"], "the man in black walks")

    def test_one_off_transformation_not_canonicalized(self):
        # Single shot with 'knight' descriptor among 4 'man in black'
        # shots — likely intentional (e.g. a flashback or one
        # surreal shot). Don't force canonicalization on a one-off.
        char_profiles = [{"id": "char_1", "physical_description": "man in black"}]
        shots = []
        for _ in range(4):
            shots.append({
                "subjects_on_screen": [{"character_id": "char_1",
                                        "visual_description": "man in black"}],
                "video_prompt": "the man in black",
            })
        shots.append({
            "subjects_on_screen": [{"character_id": "char_1",
                                    "visual_description": "knight in armor"}],
            "video_prompt": "the knight in armor",
        })
        canon = _canonicalize(shots, char_profiles)
        # Only 1 occurrence of 'knight' — below the 2-shot threshold.
        # No canonicalization.
        self.assertEqual(canon, {})

    def test_two_occurrences_triggers_canonicalization(self):
        # 2 shots with 'knight' — exactly at the threshold.
        char_profiles = [{"id": "char_1", "physical_description": "man in black"}]
        shots = [
            {"subjects_on_screen": [{"character_id": "char_1",
                                     "visual_description": "knight in armor"}],
             "video_prompt": "knight"},
            {"subjects_on_screen": [{"character_id": "char_1",
                                     "visual_description": "knight in armor"}],
             "video_prompt": "knight"},
            {"subjects_on_screen": [{"character_id": "char_1",
                                     "visual_description": "man in black"}],
             "video_prompt": "the man in black"},
        ]
        canon = _canonicalize(shots, char_profiles)
        self.assertEqual(canon, {"char_1": "knight in armor"})
        # The 'man in black' shot should now be 'knight in armor'.
        self.assertEqual(
            shots[2]["subjects_on_screen"][0]["visual_description"],
            "knight in armor",
        )
        self.assertNotIn("man in black", shots[2]["video_prompt"])

    def test_replaces_in_window_prompts_and_keyframes(self):
        char_profiles = [{"id": "char_1", "physical_description": "man in black"}]
        shots = [
            {"subjects_on_screen": [{"character_id": "char_1",
                                     "visual_description": "knight in silver armor"}],
             "video_prompt": "v1"},
            {"subjects_on_screen": [{"character_id": "char_1",
                                     "visual_description": "knight in silver armor"}],
             "video_prompt": "v2"},
            {"subjects_on_screen": [{"character_id": "char_1",
                                     "visual_description": "man in black"}],
             "video_prompt": "the man in black approaches",
             "window_prompts": ["W1: the man in black", "W2: he speaks"],
             "keyframe_prompts": ["KF1: the man in black"]},
        ]
        canon = _canonicalize(shots, char_profiles)
        self.assertEqual(canon, {"char_1": "knight in silver armor"})
        # Last shot's window_prompts and keyframe_prompts should be updated
        self.assertEqual(shots[2]["window_prompts"][0], "W1: the knight in silver armor")
        self.assertEqual(shots[2]["window_prompts"][1], "W2: he speaks")  # untouched
        self.assertEqual(shots[2]["keyframe_prompts"][0], "KF1: the knight in silver armor")

    def test_multiple_characters_canonicalized_independently(self):
        char_profiles = [
            {"id": "char_1", "physical_description": "man in black"},
            {"id": "char_2", "physical_description": "woman in red"},
        ]
        shots = [
            {"subjects_on_screen": [
                {"character_id": "char_1", "visual_description": "knight in silver armor"},
                {"character_id": "char_2", "visual_description": "queen in emerald gown"},
            ], "video_prompt": "scene 1"},
            {"subjects_on_screen": [
                {"character_id": "char_1", "visual_description": "knight in silver armor"},
                {"character_id": "char_2", "visual_description": "queen in emerald gown"},
            ], "video_prompt": "scene 2"},
            {"subjects_on_screen": [
                {"character_id": "char_1", "visual_description": "man in black"},
                {"character_id": "char_2", "visual_description": "woman in red"},
            ], "video_prompt": "the man in black bows to the woman in red"},
        ]
        canon = _canonicalize(shots, char_profiles)
        self.assertEqual(canon, {
            "char_1": "knight in silver armor",
            "char_2": "queen in emerald gown",
        })
        # Both characters fixed in last shot's text
        self.assertNotIn("man in black", shots[2]["video_prompt"])
        self.assertNotIn("woman in red", shots[2]["video_prompt"])
        self.assertIn("knight in silver armor", shots[2]["video_prompt"])
        self.assertIn("queen in emerald gown", shots[2]["video_prompt"])

    def test_no_char_profiles_no_op(self):
        # No characters defined → nothing to canonicalize.
        shots = [
            {"subjects_on_screen": [{"character_id": "char_1",
                                     "visual_description": "knight in armor"}]}
        ]
        canon = _canonicalize(shots, [])
        self.assertEqual(canon, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
