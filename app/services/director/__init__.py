# Director Mode - Layered architecture for AI-driven creative workflows
#
# Layer 1: Skill Planners   - creative planning logic per skill type
# Layer 2: Shared Schema    - canonical ShotPlan / ProductionPlan types
# Layer 3: Render Adapters  - mode-specific prompt renderers (t2v, i2v, a2v, etc.)
# Layer 4: Validators       - policy enforcement and prompt compression
