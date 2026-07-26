# B2_search_dog_scarf

![Query → Tool Call / Observation → Answer](paper_card.jpg)

## Query

What text appears on the dog’s scarf?

## Key tool calls / observations

1. `crop(scene, dog)`
   - Observation: The dog and its blue checkered scarf become visible.
2. `crop(dog_crop, neck_label)`
   - Observation: The red label resolves into readable white text.

## Answer

**“FILTHY.”**

## Figure-ready assets

- Panel 1, Cluttered scene: `panel_1_cluttered_scene.jpg`
- Panel 2, Dog crop: `panel_2_dog_crop.jpg`
- Panel 3, Text evidence: `panel_3_text_evidence.jpg`

## Provenance (for audit only)

- Final OpenVisTool source: `dataset/OpenVisTool/VisualSearch_2k.jsonl` line 1823 (1-based)
- Record SHA-256: `b847b6c524313ed041ea22ea4812827519243acce95198ceec4836a3d222965a`
- Full record: `record.jsonl` / `record.pretty.json`
- Original rollout: `source_session.jsonl`
- Full readable trajectory: `trajectory.md`
