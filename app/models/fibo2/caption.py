# Copyright 2026 Bria AI and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import math

VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"

class Caption:

    @staticmethod
    def _parse_prompt(prompt: str) -> tuple[dict | str | None, str | None]:
        prompt = "\n".join(line for line in prompt.splitlines() if not line.lstrip().startswith("#"))
        try:
            caption = json.loads(prompt)
        except json.JSONDecodeError:
            return (prompt, None)
        if not isinstance(caption, dict):
            return (prompt, None)
        instruction = caption.pop('edit_instruction', None)
        if 'structured_caption' in caption:
            caption = caption.pop('structured_caption')
            if not isinstance(caption, dict):
                return (caption or None, instruction)

        def empty(value):
            if isinstance(value, float):
                return math.isnan(value)
            return value is None or (isinstance(value, (str, dict, list)) and (not value))

        def clean(value):
            if isinstance(value, dict):
                value = {k: clean(v) for k, v in value.items()}
                return {k: v for k, v in value.items() if not empty(v)}
            if isinstance(value, list):
                return [v for v in map(clean, value) if not empty(v)]
            return value
        keys = set(caption)
        scores = ('aesthetic_score', 'preference_score')
        caption = {key: {k: v for k, v in value.items() if k not in scores} if isinstance(value, dict) else value for key, value in clean(caption).items() if key not in scores}
        caption = {key: value for key, value in caption.items() if value != {}}
        if keys == {'short_description', 'text_render'}:
            caption = {'short_description': caption.get('short_description') or '', 'text_render': caption.get('text_render') or []}
        return (caption or None, instruction)

    @staticmethod
    def _mark_text_images(caption: dict, marker: str) -> dict:
        for entry in caption.get('text_render') or []:
            if isinstance(entry, dict) and isinstance(entry.get('text'), str) and entry['text']:
                entry['image_tokens'] = marker
        return caption

    @classmethod
    def _get_edit_prompt(cls, prompt: str, num_images: int, text_images: bool=False) -> str:
        caption, instruction = cls._parse_prompt(prompt)
        if instruction is None:
            caption, instruction = (None, prompt)
        if text_images and isinstance(caption, dict):
            cls._mark_text_images(caption, VISION_MARKER)
        marker = VISION_MARKER
        if num_images == 1:
            edit = {'image': marker}
        else:
            edit = {f'image_{k}': marker for k in range(1, num_images + 1)}
        if caption:
            edit['structured_caption'] = caption
        edit['edit_instruction'] = instruction
        return json.dumps(edit, separators=(',', ':'), ensure_ascii=False)
