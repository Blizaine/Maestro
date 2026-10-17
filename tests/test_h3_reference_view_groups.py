from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from services.h3_reference_view_groups import extract_h3_reference_view_groups  # noqa: E402


def object_context(*roles):
    return "\n".join(
        f"<Picture {index}> is an object / prop reference for {role}; preserve its design."
        for index, role in enumerate(roles, start=1)
    )


class H3ReferenceViewGroupTests(unittest.TestCase):
    def test_groups_exterior_feature_and_inside_views_without_literal_same(self):
        source = (
            "Use the shuttle in reference picture1, add its solar panels like picture2, "
            "and use an inside shot from picture3."
        )
        groups = extract_h3_reference_view_groups(source, object_context(
            "shuttle exterior", "shuttle with solar panels", "Inside shuttle",
        ))

        self.assertEqual(len(groups), 1)
        group = groups[0]
        self.assertEqual(group["picture_indices"], [1, 2, 3])
        self.assertEqual(group["entity"].casefold(), "shuttle")
        self.assertEqual(group["source_evidence"], source)
        self.assertEqual(len(group["view_facts"]), 3)
        self.assertIn("one persistent shuttle", group["relationship"])
        self.assertNotIn("subject", str(group).casefold())

    def test_groups_generic_machine_exterior_feature_and_interior(self):
        source = (
            "The machine in Picture 1 has the warning light shown in Picture 2, "
            "then use Picture 3 for its interior."
        )
        groups = extract_h3_reference_view_groups(source, object_context(
            "machine exterior", "machine with warning light", "machine interior",
        ))

        self.assertEqual([group["picture_indices"] for group in groups], [[1, 2, 3]])
        self.assertEqual(groups[0]["entity"].casefold(), "machine")

    def test_role_only_inside_view_can_join_when_source_connects_it_to_object(self):
        source = "A portable generator in Picture 1 has the marked panel shown in Picture 2; use Picture 3 for an inside view."
        groups = extract_h3_reference_view_groups(source, object_context(
            "portable generator exterior", "portable generator with marked panel", "Inside",
        ))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["picture_indices"], [1, 2, 3])
        self.assertEqual(groups[0]["entity"].casefold(), "generator")

    def test_adjacent_sentences_preserve_an_explicit_object_chain(self):
        source = (
            "The shuttle in reference picture 1 hovers briefly. "
            "And then it blasts off with four jets like in picture 2. "
            "Camera switches to an inside shot using picture 3, with its pilot at the controls."
        )
        context = object_context("shuttle exterior", "shuttle with four jets", "Inside")
        groups = extract_h3_reference_view_groups(source, context)

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["picture_indices"], [1, 2, 3])
        self.assertEqual(groups[0]["source_evidence"], source)

    def test_three_sentence_object_chain_ignores_interleaved_scene_and_identity_refs(self):
        source = (
            "The shuttle in reference Picture 1 waits beside the launch tower from Picture 6. "
            "And then it lifts off with four thrusters like in Picture 2. "
            "The camera switches to an inside shot using Picture 3, with its pilot from Picture 4 at the controls."
        )
        context = (
            "<Picture 1> is an object / prop reference for shuttle exterior; preserve its design. "
            "<Picture 2> is an object / prop reference for shuttle with four thrusters; preserve its design. "
            "<Picture 3> is an object / prop reference for Inside shuttle; preserve its design. "
            "<Picture 4> is an identity reference for the pilot. "
            "<Picture 6> is a scene reference for the launch tower."
        )

        groups = extract_h3_reference_view_groups(source, context)
        self.assertEqual([group["picture_indices"] for group in groups], [[1, 2, 3]])
        self.assertEqual(groups[0]["source_evidence"], source)
        self.assertNotIn(4, groups[0]["picture_indices"])
        self.assertNotIn(6, groups[0]["picture_indices"])

    def test_adjacent_sentences_reject_new_objects_and_plural_instances(self):
        context = object_context("shuttle exterior", "shuttle with jets", "Inside")
        sources = (
            "The shuttle in Picture 1 lifts off. A newly built machine with jets follows Picture 2. "
            "Camera switches to an inside shot using Picture 3.",
            "Two shuttles appear in Picture 1. And then it blasts off with four jets like Picture 2. "
            "Camera switches to an inside shot using Picture 3.",
            "The shuttle appears in Picture 1. A similar shuttle shows its jets in Picture 2. "
            "Camera switches to an inside shot using Picture 3.",
        )
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(extract_h3_reference_view_groups(source, context), [])

    def test_explicit_co_reference_is_sufficient_without_magic_word_requirements(self):
        source = "Picture 1 shows a hand-built machine, and Picture 2 shows its interior."
        groups = extract_h3_reference_view_groups(source, object_context(
            "hand-built machine exterior", "interior of the hand-built machine",
        ))

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["picture_indices"], [1, 2])

    def test_explicitly_different_objects_are_never_merged(self):
        sources = (
            "Picture 1 shows one shuttle; Picture 2 shows a different shuttle with the same markings.",
            "A shuttle in Picture 1 and another shuttle in Picture 2 share a design.",
            "Two shuttles use the exterior in Picture 1 and the interior in Picture 2.",
            "A shuttle in Picture 1 and a shuttle in Picture 2 show the same paint.",
        )
        context = object_context("shuttle exterior", "shuttle with solar panels")
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(extract_h3_reference_view_groups(source, context), [])

    def test_separate_named_object_chains_in_one_sentence_are_not_grouped(self):
        source = (
            "The shuttle from Picture 1 has solar panels like Picture 2, "
            "and the machine interior is shown in Picture 3."
        )
        context = object_context(
            "shuttle exterior", "shuttle with solar panels", "machine interior",
        )

        self.assertEqual(extract_h3_reference_view_groups(source, context), [])

        bare_inside_context = object_context(
            "shuttle exterior", "shuttle with solar panels", "Inside",
        )
        self.assertEqual(extract_h3_reference_view_groups(source, bare_inside_context), [])

    def test_merely_similar_or_unconnected_images_are_not_grouped(self):
        context = object_context("shuttle exterior", "shuttle with solar panels")
        sources = (
            "Picture 1 shows a shuttle. A similar shuttle with panels appears in Picture 2.",
            "A shuttle and a picture of solar panels are shown in Picture 1 and Picture 2.",
            "Picture 1 and Picture 2 show shuttle designs in separate scenes.",
        )
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(extract_h3_reference_view_groups(source, context), [])

    def test_only_mentioned_canonical_object_picture_slots_are_returned(self):
        source = "The shuttle in Picture 1 has the solar panels from Picture 2."
        context = object_context(
            "shuttle exterior", "shuttle with solar panels", "shuttle interior",
        )
        groups = extract_h3_reference_view_groups(source, context)

        self.assertEqual([group["picture_indices"] for group in groups], [[1, 2]])
        self.assertNotIn(3, groups[0]["picture_indices"])

    def test_non_object_roles_do_not_join_object_view_groups(self):
        source = "The shuttle in Picture 1 has the interior from Picture 2."
        context = (
            "<Picture 1> is an object / prop reference for shuttle exterior; preserve its design.\n"
            "<Picture 2> is a composition and blocking reference for shuttle interior."
        )

        self.assertEqual(extract_h3_reference_view_groups(source, context), [])

    def test_canonical_declarations_may_share_a_line(self):
        source = "The shuttle in Picture 1 has its interior from Picture 2."
        context = object_context("shuttle exterior", "shuttle interior").replace("\n", " ")

        groups = extract_h3_reference_view_groups(source, context)
        self.assertEqual([group["picture_indices"] for group in groups], [[1, 2]])

    def test_canonical_object_slots_above_nine_remain_valid(self):
        source = "The machine in Picture 10 has its interior from Picture 11."
        context = (
            "<Picture 10> is an object / prop reference for machine exterior; preserve its design. "
            "<Picture 11> is an object / prop reference for machine interior; preserve its design."
        )

        groups = extract_h3_reference_view_groups(source, context)
        self.assertEqual([group["picture_indices"] for group in groups], [[10, 11]])

    def test_ambiguous_duplicate_or_missing_object_context_fails_closed(self):
        source = "The shuttle in Picture 1 has its interior from Picture 2."
        self.assertEqual(extract_h3_reference_view_groups(source, ""), [])
        self.assertEqual(extract_h3_reference_view_groups(source, "<Picture 1> is a scene reference."), [])
        duplicate_context = object_context("shuttle exterior", "shuttle interior") + (
            "\n<Picture 2> is an object / prop reference for another shuttle."
        )
        self.assertEqual(extract_h3_reference_view_groups(source, duplicate_context), [])

    def test_noncanonical_indices_and_oversized_inputs_fail_closed(self):
        context = object_context("shuttle exterior", "shuttle interior")
        self.assertEqual(
            extract_h3_reference_view_groups("The shuttle in picture 01 has an interior in Picture 2.", context),
            [],
        )
        self.assertEqual(extract_h3_reference_view_groups("x" * 30_001, context), [])


if __name__ == "__main__":
    unittest.main()
