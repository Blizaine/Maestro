"""Window rendering keeps global planning history out of local visual direction."""
from copy import deepcopy
import unittest

from services.h3_sequence_planner import compile_h3_reference_sequence_prompts
from services.h3_window_context import stable_h3_window_context


class H3WindowContextTests(unittest.TestCase):
    def compile(self, plan, reference_relationships='<Subject 1> is Ada from <Picture 1>.'):
        return compile_h3_reference_sequence_prompts(
            plan, [{'index': index+1, 'frames': 240, 'duration_seconds': 10.0}
                   for index in range(len(plan['clips']))],
            reference_relationships=reference_relationships,
            default_retention='<Subject 1>: fully_preserved', task_types='reference generation',
        )

    def plan(self):
        return {'source_prompt': 'Ada boards a transport, crosses the bay, then arrives at the museum.',
                'setting_continuity': 'The setting transitions from the dock, to open water, to the museum.',
                'clips': [
                    {'opening_state': 'Ada stands at the dock.',
                     'closing_state': 'Continue from the visible result of: The transport launches. Do not repeat it.',
                     'native_closing_state': 'Ada is seated aboard the transport, afloat beyond the dock.',
                     'shots': [{'start_seconds': 0, 'end_seconds': 10,
                                'action': 'Ada boards and the transport departs from the dock.'}]},
                    {'opening_state': 'Continue from the visible result of: The transport launches. Do not repeat it.',
                     'closing_state': 'Continue from the visible result of: The transport crosses the bay. Do not repeat it.',
                     'native_closing_state': 'Ada remains seated aboard the transport in open water.',
                     'shots': [{'start_seconds': 0, 'end_seconds': 10,
                                'action': 'The transport crosses open water with Ada already seated aboard.'}]},
                    {'opening_state': 'Continue from the visible result of: The transport crosses the bay. Do not repeat it.',
                     'closing_state': 'Ada stands at the museum entrance.',
                     'shots': [{'start_seconds': 0, 'end_seconds': 10,
                                'action': 'The transport docks at the museum and Ada disembarks.'}]},
                ]}

    def test_itinerary_is_not_repeated_in_transit_window(self):
        plan=self.plan(); original=deepcopy(plan)
        windows=self.compile(plan)
        self.assertNotIn('museum', windows[1]['prompt'])
        self.assertNotIn('transitions from', windows[1]['prompt'])
        self.assertIn('crosses open water', windows[1]['prompt'])
        self.assertIn('museum', windows[2]['prompt'])
        self.assertEqual(plan,original)

    def test_style_retains_aesthetic_without_future_location_itinerary(self):
        plan = self.plan()
        plan['visual_style'] = (
            'Naturalistic lighting and rich textures. '
            'Lighting transitions from the dock to the bright museum hall. '
            'Composition emphasizes readable movement.'
        )
        windows = self.compile(plan)
        self.assertIn('naturalistic lighting and rich textures', windows[1]['prompt'])
        self.assertIn('Composition emphasizes readable movement', windows[1]['prompt'])
        self.assertNotIn('museum', windows[1]['prompt'])
        self.assertIn('Lighting transitions', plan['visual_style'])

    def test_generic_opening_place_cues_are_removed_but_static_setting_is_kept(self):
        opening_cues = (
            'The sequence begins at a quiet rail terminal.',
            'The story starts inside a glass atrium.',
            'The journey opens beside a mountain lodge.',
            'The video begins on a stone courtyard.',
            'The initial setting is an underground archive.',
            'The opening location: a riverside workshop.',
            'The first setting is a sunlit conservatory.',
            'The location at the start is the lower hall.',
        )
        for opening in opening_cues:
            with self.subTest(opening=opening):
                stable = stable_h3_window_context(
                    opening + ' A shared architectural style remains consistent.'
                )
                self.assertNotIn(opening, stable)
                self.assertIn('A shared architectural style remains consistent', stable)

    def test_shift_routes_are_removed_while_static_aesthetics_and_effect_origins_remain(self):
        setting = stable_h3_window_context(
            'A consistent interior remains. The location shifts from the entry hall to the upper landing.'
        )
        self.assertIn('A consistent interior remains', setting)
        self.assertNotIn('entry hall', setting)
        self.assertNotIn('upper landing', setting)

        visual = stable_h3_window_context(
            'Lighting is bright and dynamic, shifting from cool studio illumination at the opening '
            'to clear daylight later. Naturalistic textures and restrained colors remain consistent. '
            'A beam travels from the emitter to the target.',
            visual_style=True,
        )
        self.assertIn('Lighting is bright and dynamic', visual)
        self.assertIn('Naturalistic textures and restrained colors remain consistent', visual)
        self.assertIn('A beam travels from the emitter to the target', visual)
        self.assertNotIn('cool studio illumination', visual)
        self.assertNotIn('clear daylight later', visual)

    def test_native_compiler_filters_global_opening_place_and_lighting_route(self):
        plan = self.plan()
        plan['setting_continuity'] = (
            'The journey opens at a central rail terminal. '
            'A shared transit hub remains consistent throughout.'
        )
        plan['visual_style'] = (
            'Lighting is bright and dynamic, shifting from warm interior illumination in opening shots '
            'to natural daylight in later exterior shots. Soft film grain remains constant.'
        )
        windows = self.compile(plan)
        for window in windows:
            with self.subTest(window=window['index']):
                self.assertNotIn('central rail terminal', window['prompt'])
                self.assertNotIn('warm interior illumination', window['prompt'])
                self.assertNotIn('natural daylight in later exterior shots', window['prompt'])
                self.assertIn('a shared transit hub remains consistent throughout', window['prompt'].casefold())
                self.assertIn('lighting is bright and dynamic', window['prompt'].casefold())
                self.assertIn('soft film grain remains constant', window['prompt'].casefold())

    def test_native_snapshot_replaces_action_recap_without_rewriting_receipts(self):
        plan=self.plan(); windows=self.compile(plan)
        self.assertIn('afloat beyond the dock', windows[1]['opening_state'])
        self.assertIn('remains seated', windows[2]['opening_state'])
        self.assertNotIn('Continue from the visible result', windows[1]['prompt'])
        self.assertNotIn('The transport launches', windows[1]['prompt'])
        self.assertIn('Continue from the visible result',plan['clips'][0]['closing_state'])

    def test_legacy_recap_has_safe_visual_continuation_without_action_inference(self):
        plan=self.plan()
        for clip in plan['clips']:clip.pop('native_closing_state',None)
        windows=self.compile(plan)
        self.assertNotIn('The transport launches',windows[1]['opening_state'])
        self.assertNotIn('Do not repeat',windows[1]['prompt'])
        self.assertIn('visible outcome',windows[1]['opening_state'])
        self.assertEqual(windows[1]['shots'][0]['action'],plan['clips'][1]['shots'][0]['action'])

    def test_explicit_independent_cut_opening_is_preserved(self):
        plan=self.plan()
        plan['clips'][1]['opening_state']='Independent cut: Bea is seated inside a workshop.'
        windows=self.compile(plan)
        self.assertEqual(windows[1]['opening_state'],'Independent cut: Bea is seated inside a workshop')

    def test_static_setting_and_window_local_setting_are_retained(self):
        plan=self.plan()
        plan['setting_continuity']='A historic bathhouse from the 1920s with access to a roof terrace.'
        plan['clips'][1]['setting_continuity']='The enclosed rooftop pool.'
        windows=self.compile(plan)
        self.assertIn('historic bathhouse',windows[0]['prompt'])
        self.assertIn('enclosed rooftop pool',windows[1]['prompt'])
        self.assertNotIn('historic bathhouse',windows[1]['prompt'])

    def test_single_window_setting_keeps_requested_transitions(self):
        plan=self.plan();plan['clips']=plan['clips'][:1]
        window=self.compile(plan)[0]
        self.assertIn('transitions from',window['prompt'])

    def test_source_linked_object_views_are_carried_without_new_subject_ids(self):
        plan=self.plan()
        plan['source_prompt']=(
            'Ada operates the machine in picture 1. It uses the panel shown in picture 2. '
            'The camera switches to an inside shot using picture 3.'
        )
        context='\n'.join([
            '<Picture 1> is an object / prop reference for machine; preserve its design.',
            '<Picture 2> is an object / prop reference for machine with panel; preserve its design.',
            '<Picture 3> is an object / prop reference for inside machine; preserve its design.',
            '<Subject 1> is Ada from <Picture 4>.',
        ])
        windows=self.compile(plan,context)
        for window in windows:
            self.assertEqual(window['reference_view_groups'][0]['picture_indices'],[1,2,3])
            self.assertIn('views of one persistent machine',window['prompt'])
            self.assertNotIn('<Subject 2>',window['prompt'])


if __name__=='__main__':unittest.main()
