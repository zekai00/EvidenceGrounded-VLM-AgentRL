"""Prompt construction for EvidenceGrounded rollout policies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


IMAGE_SIZE_CACHE: dict[str, tuple[int, int]] = {}
CLAIM_FIELD_SPEC = (
    "caption_text|image_scope|depicted_work_title|displayed_region|object_type|artist|dynasty|"
    "visual_elements|technique|composition|medium_dimensions|collection"
)
EVIDENCE_POLICY_KEYS = [
    "clean_evidence_type",
    "adjudicated_evidence_role",
    "adjudication_status",
    "adjudicated_claim_allowed_fields",
    "usable_for_claim_by_adjudication",
]


@dataclass
class PromptConfig:
    max_history_actions: int = 8
    max_tool_results: int = 6
    max_evidence_per_result: int = 3
    snippet_chars: int = 180
    max_text_chars: int = 24000
    head_text_chars: int = 5000
    coordinate_info: bool = True
    tool_schema: str = "highlighted_direct"
    compact_claim_state: bool = False
    region_selection_hint: bool = True
    strict_claim_phase_hint: bool = False
    dynamic_tool_schema: bool = False
    field_policy_prompt: bool = False


def build_messages_from_observation(
    obs: dict[str, Any],
    config: PromptConfig | None = None,
    *,
    include_assistant_action: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    config = config or PromptConfig()
    content: list[dict[str, Any]] = []
    for image in obs.get("images") or []:
        path = image.get("path") if isinstance(image, dict) else image
        if path:
            content.append({"type": "image", "image": str(path)})
    content.append({"type": "text", "text": build_prompt_text(obs, config)})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if include_assistant_action is not None:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(include_assistant_action, ensure_ascii=False, separators=(",", ":")),
            }
        )
    return compact_messages(messages, config.max_text_chars, config.head_text_chars)


def build_prompt_text(obs: dict[str, Any], config: PromptConfig) -> str:
    history = [simplify_action(item) for item in (obs.get("history") or [])[-config.max_history_actions :]]
    tool_results = [
        simplify_tool_result(item, config) for item in (obs.get("tool_results") or [])[-config.max_tool_results :]
    ]
    draft_claims = obs.get("draft_claims") or []
    current_claim_state = obs.get("claim_state") or claim_state_from_claims(draft_claims)
    images = obs.get("images") or []
    if config.tool_schema == "region":
        target_line = "目标：根据无红框 PDF 原始页面、候选区域工具、局部裁剪图和可追溯证据，定位目标山水画图像并写出有证据支撑的结构化 claim。"
    elif config.tool_schema in {"evidence_select", "chunked_claim"}:
        target_line = "目标：根据 PDF 页面、候选区域、候选证据、局部裁剪图和可追溯证据，为目标山水画图像选择可信 evidence_id，并写出有证据支撑的结构化 claim。"
    elif config.tool_schema == "inspect_crop":
        target_line = "目标：先检查 PDF 页面布局，再裁剪目标山水画图像，之后检索证据并写出有证据支撑的结构化 claim。"
    elif config.tool_schema == "no_select":
        target_line = "目标：先检查 PDF 页面布局，再裁剪目标山水画图像，之后直接打开/检索可见证据并写出有证据支撑的结构化 claim；本协议不使用 select_evidence。"
    else:
        target_line = "目标：根据 PDF 页面图像、局部裁剪图和可追溯证据，为红框/目标山水画图像写出有证据支撑的结构化 claim。"
    lines = [
        "你是 evidence-grounded figure understanding 的 VLM tool-call agent。",
        target_line,
        f"task_id：{obs.get('task_id')}；step：{obs.get('step') if obs.get('step') is not None else len(obs.get('history') or [])}",
        f"source_file：{obs.get('source_file', '')}；page：{obs.get('page', '')}",
        f"输入图像：{len(images)} 张。第 1 张通常是 PDF 页面；第 2 张通常是已裁剪的目标图。",
    ]
    if obs.get("target_evidence_hints"):
        lines.extend(
            [
                "目标图注/目标证据线索（用于判断要裁剪哪一个 figure_candidate；这是任务目标说明，不是 bbox）：",
                json.dumps(
                    [simplify_evidence(item, config) for item in obs.get("target_evidence_hints") or []],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ]
        )
    phase_hints = [
        hint
        for hint in [
            region_selection_phase_hint(obs, config),
            claim_phase_hint(obs, config, current_claim_state),
        ]
        if hint
    ]
    if phase_hints:
        lines.extend(phase_hints)
    if config.field_policy_prompt:
        lines.extend(field_policy_prompt_lines(obs))
    if config.coordinate_info:
        image_info = [
            {"index": index + 1, "role": item.get("role"), "path": item.get("path"), "size": image_size(item.get("path"))}
            for index, item in enumerate(images)
            if isinstance(item, dict)
        ]
        lines.extend(
            [
                f"图像尺寸：{json.dumps(image_info, ensure_ascii=False, separators=(',', ':'))}",
                coordinate_rule(config.tool_schema),
            ]
        )
    dynamic_allowed = (obs.get("available_actions") or None) if config.dynamic_tool_schema else None
    lines.extend(tool_schema_lines(config.tool_schema, dynamic_allowed))
    if obs.get("available_actions"):
        lines.extend(
            [
                "当前阶段允许的工具（必须只从这里选择 action）：",
                json.dumps(obs.get("available_actions"), ensure_ascii=False, separators=(",", ":")),
            ]
        )
        if isinstance(obs.get("tool_mask"), dict):
            lines.append(
                "当前阶段判定："
                + json.dumps(
                    {
                        "phase": obs["tool_mask"].get("phase"),
                        "reason": obs["tool_mask"].get("reason"),
                        **(
                            {"blocked_actions": obs["tool_mask"].get("blocked_actions") or []}
                            if config.dynamic_tool_schema
                            else {}
                        ),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
    evidence_id_rule = (
        "当前可用 evidence_ids（open_evidence/write_claims_chunk/write_claim/write_claims_batch 只能使用这里或工具返回里明确出现过的 id；必须逐字符原样复制完整 id，包括所有下划线；禁止编造 ev_888、ev_xxx 等占位 id）："
        if config.tool_schema == "no_select"
        else "当前可用 evidence_ids（open_evidence/select_evidence/write_claims_chunk 只能使用这里或工具返回里明确出现过的 id；必须逐字符原样复制完整 id，包括所有下划线；禁止编造 ev_888、ev_xxx 等占位 id）："
    )
    tool_order_rule = (
        "工具顺序约束：open_evidence 不是搜索工具，只能打开当前可用 evidence_ids 中已经出现的 id；crop 后最多先打开 1 条可见 local evidence，之后应 retrieve_evidence 或 write_claims_chunk，不要反复 open_evidence；本协议禁止 select_evidence。"
        if config.tool_schema == "no_select"
        else "工具顺序约束：open_evidence 不是搜索工具，只能打开当前可用 evidence_ids 中已经出现的 id；crop 后最多先打开 1 条可见 local evidence，之后应 retrieve_evidence 或 write_claims_chunk，不要反复 open_evidence。"
    )
    lines.extend(
        [
            "约束：只输出一个 JSON 对象；不要输出 markdown；不要编造作品名、画家、朝代、技法；证据不足就 abstain；如果给出了当前阶段允许的工具，action 必须在该列表中；done 不是 action 名，完成时只能输出 {\"action\":\"finish\",\"status\":\"done\"}，且只有 finish 出现在允许工具中才可使用。",
            "证据边界约束：如果证据摘要包含 adjudicated_claim_allowed_fields，只能用该 evidence 支持这些字段；如果 usable_for_claim_by_adjudication=false 或 adjudication_status 不是 accepted_auto，应对该字段 abstain 或继续检索。",
            tool_order_rule,
            "历史动作（保留最近若干步）：",
            json.dumps(history, ensure_ascii=False, separators=(",", ":")),
            "工具返回摘要（保留最近若干条，每条检索只保留前几个候选证据）：",
            json.dumps(tool_results, ensure_ascii=False, separators=(",", ":")),
            "已选择 evidence_ids：",
            json.dumps(obs.get("selected_evidence_ids") or [], ensure_ascii=False, separators=(",", ":")),
            evidence_id_rule,
            json.dumps(obs.get("visible_evidence_ids") or [], ensure_ascii=False, separators=(",", ":")),
            "当前 claim_state（优先依据它判断已写字段和剩余字段）：",
            json.dumps(current_claim_state, ensure_ascii=False, separators=(",", ":")),
            "当前 claims 明细：",
            json.dumps([] if config.compact_claim_state else draft_claims, ensure_ascii=False, separators=(",", ":")),
            final_action_guard(obs, config, current_claim_state),
            "请根据当前状态选择下一步工具调用。只输出一个 JSON 对象。",
        ]
    )
    return "\n".join(lines)


def field_policy_prompt_lines(obs: dict[str, Any]) -> list[str]:
    remaining = list((obs.get("claim_state") or {}).get("remaining_fields") or [])
    return [
        "字段级 evidence policy probe：",
        "1. 每个非 abstain claim 必须引用能够支持该字段的 evidence_id；不能因为一个 evidence 与作品相关，就把它用于所有字段。",
        "2. local_caption 主要支持 caption_text，以及图注中明确出现的题名、作者、朝代、尺寸、馆藏；不能默认支持 image_scope、displayed_region、object_type。",
        "3. image_scope/displayed_region/object_type 只有在 evidence 文本明确说明全幅/局部/册页/细部/对象类型时才可写；否则应 retrieve/open 外部 evidence，仍不足则 abstain。",
        "4. retrieve_evidence 后，如果要用检索结果支持 claim，必须先 open_evidence 读取对应 evidence；不要只检索不打开就写 claim。",
        "当前 remaining_fields："
        + json.dumps(remaining, ensure_ascii=False, separators=(",", ":")),
    ]


def final_action_guard(obs: dict[str, Any], config: PromptConfig, current_claim_state: dict[str, Any]) -> str:
    allowed = list(obs.get("available_actions") or [])
    if not allowed:
        return "最终动作约束：未提供阶段 mask；按工具顺序选择下一步。"
    phase = (obs.get("tool_mask") or {}).get("phase") if isinstance(obs.get("tool_mask"), dict) else None
    if config.tool_schema in {"inspect_crop", "no_select"} and phase == "region_selection":
        preferred_crop = "crop_target" if "crop_target" in allowed else "crop_region"
        ranked = [
            item
            for item in (obs.get("regions") or [])
            if isinstance(item, dict) and item.get("type") == "figure_candidate" and item.get("target_region_rank") is not None
        ]
        if ranked:
            best = sorted(
                ranked,
                key=lambda item: (
                    int(item.get("target_region_rank") or 999),
                    -float(item.get("target_region_sort_score") or 0.0),
                ),
            )[0]
            return (
                "最终动作约束：当前阶段只允许裁剪目标图像，禁止检索、打开证据、写 claim 或 finish。"
                f"本步必须输出 {{\"action\":\"{preferred_crop}\",\"region_id\":\"{best.get('region_id')}\"}}。"
                "这个 region_id 来自 target_region_rank=1 的 figure_candidate，"
                f"target_caption_match_score={best.get('target_caption_match_score')}，"
                f"caption_link_score={best.get('caption_link_score')}。"
            )
        return (
            "最终动作约束：当前阶段只允许裁剪目标图像，禁止检索、打开证据、写 claim 或 finish。"
            f"本步必须输出 {{\"action\":\"{preferred_crop}\",\"region_id\":\"候选figure_candidate的region_id\"}}。"
        )
    if phase in {"claim_writing", "claim_continuation"}:
        remaining = current_claim_state.get("remaining_fields") or []
        if remaining:
            return (
                "最终动作约束：当前阶段只允许继续写 claim 或 abstain。"
                "优先调用 write_claims_chunk，一次只写 1 个 remaining_fields 中的字段；"
                "也可以用 write_claim 写单个字段。visual_elements、composition 等长字段必须简短，不要输出重复长数组。"
            )
        return "最终动作约束：当前字段已经完成，本步应输出 {\"action\":\"finish\",\"status\":\"done\"}。"
    if phase in {"local_evidence_opening", "evidence_retrieval_after_open", "evidence_opening"} and "retrieve_evidence" in allowed:
        return (
            "最终动作约束：本步如果调用 retrieve_evidence，query 必须短且可闭合为合法 JSON："
            "query 长度不超过 80 个汉字/英文词，禁止重复同一个词组或作品名，禁止把整段证据复制进 query；"
            "优先用 source_file 短标题 + 当前图注关键词 + 山水画/桥梁/构图等 1-3 个检索词。"
            "如果当前阶段还允许 write_claims_chunk 且本地 evidence 已足够，也可以直接写 claim；"
            "无论选择哪个 action，都必须输出完整闭合的单个 JSON 对象。"
        )
    if len(allowed) == 1:
        return f"最终动作约束：当前阶段唯一允许的 action 是 {allowed[0]}，输出其他 action 会被判为非法。"
    return "最终动作约束：本步 action 必须属于当前阶段允许的工具列表：" + json.dumps(
        allowed, ensure_ascii=False, separators=(",", ":")
    )


def claim_phase_hint(obs: dict[str, Any], config: PromptConfig, current_claim_state: dict[str, Any]) -> str:
    if config.tool_schema not in {"chunked_claim", "inspect_crop", "no_select"}:
        return ""
    phase = (obs.get("tool_mask") or {}).get("phase") if isinstance(obs.get("tool_mask"), dict) else None
    remaining = current_claim_state.get("remaining_fields") or []
    if config.strict_claim_phase_hint and phase in {"claim_writing", "claim_continuation"}:
        if remaining:
            blocked_tools = "open_evidence、retrieve_evidence" if config.tool_schema == "no_select" else "open_evidence、retrieve_evidence 或 select_evidence"
            extra = ""
            if config.field_policy_prompt:
                extra = (
                    "对于 image_scope、displayed_region、object_type，如果当前只有 local_caption 且图注没有明确支持，"
                    "本阶段应对该字段 abstain，不要用 local_caption 硬写。"
                )
            return (
                f"阶段提示：当前已经进入 claim 写入阶段，当前阶段禁止继续 {blocked_tools}。"
                "如果 current_claim_state.remaining_fields 仍有字段，下一步必须优先调用 write_claims_chunk，"
                "一次只写 1 个尚未写过的字段；也可以用 write_claim 写单个字段。"
                "证据不足的字段用 abstain_claim 或 write_claims_chunk.abstains。"
                "不要重复已经在 written_fields 或 abstained_fields 中出现的字段；复杂字段必须简短，避免超长 JSON。"
                + extra
            )
        return "阶段提示：当前 claim 字段已经写完或 abstain 完成，下一步必须调用 finish。"
    if not remaining:
        return "阶段提示：claim_state 显示目标字段已经写完或 abstain 完成，优先 finish。"
    selected_ids = obs.get("selected_evidence_ids") or []
    history_actions = [item.get("action") for item in obs.get("history") or [] if isinstance(item, dict)]
    tool_names = [item.get("tool") for item in obs.get("tool_results") or [] if isinstance(item, dict)]
    has_evidence = bool(selected_ids) or "select_evidence" in tool_names or "open_evidence" in tool_names
    has_prepared_image = any(tool in tool_names for tool in ["crop_region", "crop_target", "crop_image"])
    has_retrieved = "retrieve_evidence" in history_actions or "retrieve_evidence" in tool_names
    if has_evidence and (has_prepared_image or has_retrieved):
        evidence_phrase = "已有 open_evidence 或工具返回中的 evidence" if config.tool_schema == "no_select" else "已有 selected_evidence_ids 或工具返回中的 evidence"
        extra = ""
        if config.field_policy_prompt:
            extra = "写每个字段前检查 claim_use_hint/adjudicated_claim_allowed_fields；非 caption 字段不能默认复用 local_caption。"
        return (
            "阶段提示：当前通常已经完成候选区域选择、裁剪、证据检索和证据打开；"
            f"如果 claim_state 显示仍有待写字段，并且{evidence_phrase}，"
            "优先调用 write_claims_chunk 写入下一组结构化 claim。"
            "每次只写 1 个字段，复杂字段必须简短。除非明确缺少必要证据，否则不要继续 open_evidence 或 finish。"
            + extra
        )
    return ""


def region_selection_phase_hint(obs: dict[str, Any], config: PromptConfig) -> str:
    if not config.region_selection_hint:
        return ""
    if config.tool_schema not in {"region", "evidence_select", "chunked_claim", "inspect_crop", "no_select"}:
        return ""
    history_actions = [item.get("action") for item in obs.get("history") or [] if isinstance(item, dict)]
    tool_names = [item.get("tool") for item in obs.get("tool_results") or [] if isinstance(item, dict)]
    has_regions = (
        bool(obs.get("available_region_ids") or obs.get("regions"))
        or "inspect_page" in history_actions
        or "inspect_page" in tool_names
        or "propose_regions" in history_actions
        or "propose_regions" in tool_names
    )
    has_crop = any(tool in tool_names for tool in ["crop_region", "crop_target", "crop_image"])
    if has_regions and not has_crop:
        allowed = set(obs.get("available_actions") or [])
        phase = (obs.get("tool_mask") or {}).get("phase") if isinstance(obs.get("tool_mask"), dict) else None
        preferred_crop = "crop_target" if "crop_target" in allowed else "crop_region"
        if phase == "region_selection" and allowed <= {"crop_target", "crop_region"}:
            return (
                f"阶段提示：当前已经通过 inspect_page 看到页面布局区域；当前阶段只允许裁剪目标区域，优先使用 {preferred_crop}，"
                "禁止 select_evidence、open_evidence、retrieve_evidence 或 finish。"
                f"下一步必须输出 {{\"action\":\"{preferred_crop}\",\"region_id\":\"r0\"}} 这种 JSON，"
                "region_id 必须来自候选区域列表。正确目标图像不一定是第一个候选；不要裁剪正文、页眉页脚或图注框本身，"
                "优先选择与任务目标、图注内容和页面中的山水画图像区域一致的 figure_candidate。"
                "如果候选区域里出现 target_region_rank，必须优先裁剪 target_region_rank 最小的 figure_candidate；"
                "排序依据是目标图注文本匹配分数 target_caption_match_score，其次是图像与图注的几何关联分数 caption_link_score。"
            )
        return (
            "阶段提示：当前已经通过 inspect_page 看到页面布局区域；如果下一步要裁剪目标图像，"
            "请调用 crop_target。正确目标图像不一定是第一个候选；不要裁剪 text_or_caption_candidate、"
            "正文、页眉页脚或图注框本身；优先选择与任务目标、图注内容和页面中的山水画图像区域一致的 figure_candidate。"
            "如果候选区域里出现 target_region_rank，必须优先裁剪 target_region_rank 最小的 figure_candidate；"
            "排序依据是目标图注文本匹配分数 target_caption_match_score，其次是图像与图注的几何关联分数 caption_link_score。"
        )
    return ""


def tool_schema_lines(schema: str, allowed_actions: list[str] | None = None) -> list[str]:
    allowed = set(allowed_actions or [])

    def keep(action: str) -> bool:
        return not allowed or action in allowed

    if schema in {"inspect_crop", "no_select"}:
        rows = [
            ("inspect_page", '{"action":"inspect_page","top_k":整数}'),
            ("crop_target", '{"action":"crop_target","region_id":"r_xxx"} 或 {"action":"crop_target","bbox":[x1,y1,x2,y2]}'),
            (
                "retrieve_evidence",
                '{"action":"retrieve_evidence","query":"...","scope":"current_page|nearby_pages|same_document|corpus","anchor":{"source_file":"...","page":页码,"bbox":[x1,y1,x2,y2]},"top_k":整数}',
            ),
            ("open_evidence", '{"action":"open_evidence","evidence_id":"ev_xxx"}'),
            (
                "write_claims_chunk",
                f'{{"action":"write_claims_chunk","claims":[{{"field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["从当前可用evidence_ids原样复制"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}',
            ),
            (
                "write_claim",
                f'{{"action":"write_claim","field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["从当前可用evidence_ids原样复制"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}',
            ),
            ("abstain_claim", '{"action":"abstain_claim","field":"字段名","reason":"证据不足原因"}'),
            (
                "write_claims_batch",
                f'{{"action":"write_claims_batch","claims":[{{"field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["从当前可用evidence_ids原样复制"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}',
            ),
            ("finish", '{"action":"finish","status":"done"}'),
        ]
        lines = ["当前可用工具格式示例："]
        filtered = [text for action, text in rows if keep(action)]
        lines.extend(f"{idx}. {text}" for idx, text in enumerate(filtered, start=1))
        if schema == "no_select":
            lines.append("工具顺序：inspect_page -> crop_target -> open_evidence(可见 local_caption_id，可选) -> retrieve_evidence -> open_evidence(检索结果，可选) -> write_claims_chunk/write_claim/abstain_claim -> finish；禁止 select_evidence。")
        else:
            lines.append("工具顺序：inspect_page -> crop_target -> retrieve/open/select evidence -> write_claims_chunk/write_claim/abstain_claim -> finish。")
        return lines
    if schema == "chunked_claim":
        return [
            "可用工具：",
            '1. {"action":"inspect_page"}',
            '2. {"action":"propose_regions","top_k":整数}',
            '3. {"action":"select_evidence","evidence_ids":["ev_xxx或local_caption_xxx"]}',
            '4. {"action":"crop_region","region_id":"r_xxx"}',
            '5. {"action":"retrieve_evidence","query":"...","scope":"current_page|nearby_pages|same_document|corpus","anchor":{"source_file":"...","page":页码,"bbox":[x1,y1,x2,y2]},"top_k":整数}',
            '6. {"action":"open_evidence","evidence_id":"ev_xxx"}',
            f'7. {{"action":"write_claims_chunk","claims":[{{"field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}',
            f'8. {{"action":"write_claim","field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}',
            '9. {"action":"abstain_claim","field":"字段名","reason":"证据不足原因"}',
            f'10. {{"action":"write_claims_batch","claims":[{{"field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}',
            '11. {"action":"finish","status":"done"}',
            "写 claim 阶段优先使用 write_claims_chunk；每次只写 1 个字段；也可以用 write_claim 写单个字段；不要重复已经在 claim_state.written_fields 或 abstained_fields 中出现的字段。",
        ]
    if schema == "evidence_select":
        return [
            "可用工具：",
            '1. {"action":"inspect_page"}',
            '2. {"action":"propose_regions","top_k":整数}',
            '3. {"action":"select_evidence","evidence_ids":["ev_xxx或local_caption_xxx"]}',
            '4. {"action":"crop_region","region_id":"r_xxx"}',
            '5. {"action":"retrieve_evidence","query":"...","scope":"current_page|nearby_pages|same_document|corpus","anchor":{"source_file":"...","page":页码,"bbox":[x1,y1,x2,y2]},"top_k":整数}',
            '6. {"action":"open_evidence","evidence_id":"ev_xxx"}',
            f'7. {{"action":"write_claim","field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}',
            '8. {"action":"abstain_claim","field":"字段名","reason":"证据不足原因"}',
            f'9. {{"action":"write_claims_batch","claims":[{{"field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}',
            '10. {"action":"finish","status":"done"}',
        ]
    if schema == "region":
        return [
            "可用工具：",
            '1. {"action":"inspect_page"}',
            '2. {"action":"propose_regions","top_k":整数}',
            '3. {"action":"crop_region","region_id":"r_xxx"}',
            '4. {"action":"retrieve_evidence","query":"...","scope":"current_page|nearby_pages|same_document|corpus","anchor":{"source_file":"...","page":页码,"bbox":[x1,y1,x2,y2]},"top_k":整数}',
            '5. {"action":"open_evidence","evidence_id":"ev_xxx"}',
            f'6. {{"action":"write_claim","field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}',
            '7. {"action":"abstain_claim","field":"字段名","reason":"证据不足原因"}',
            f'8. {{"action":"write_claims_batch","claims":[{{"field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}',
            '9. {"action":"finish","status":"done"}',
        ]
    return [
        "可用工具：",
        '1. {"action":"crop_image","bbox":[x1,y1,x2,y2]}',
        '2. {"action":"retrieve_evidence","query":"...","scope":"current_page|nearby_pages|same_document|corpus","anchor":{"source_file":"...","page":页码,"bbox":[x1,y1,x2,y2]},"top_k":整数}',
        '3. {"action":"open_evidence","evidence_id":"ev_xxx"}',
        f'4. {{"action":"write_claim","field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}',
        '5. {"action":"abstain_claim","field":"字段名","reason":"证据不足原因"}',
        f'6. {{"action":"write_claims_batch","claims":[{{"field":"{CLAIM_FIELD_SPEC}","value":值,"evidence_ids":["ev_xxx"],"visual_bbox":[x1,y1,x2,y2]或null,"confidence":0到1}}],"abstains":[{{"field":"字段名","reason":"证据不足原因"}}]}}',
        '7. {"action":"finish","status":"done"}',
    ]


def coordinate_rule(schema: str) -> str:
    if schema == "inspect_crop":
        return (
            "坐标规则：inspect_page 会返回 layout regions；region bbox 和 crop_target 的 bbox 都使用第 1 张 PDF 原始页面图像的像素坐标，原点在左上角，格式为 [x1,y1,x2,y2]；"
            "当前页面没有红框，先用 inspect_page 检查页面布局，再用 crop_target 裁剪目标图像；下一步工具必须服从“当前阶段允许的工具”列表。"
        )
    if schema == "no_select":
        return (
            "坐标规则：inspect_page 会返回 layout regions；region bbox 和 crop_target 的 bbox 都使用第 1 张 PDF 原始页面图像的像素坐标，原点在左上角，格式为 [x1,y1,x2,y2]；"
            "当前页面没有红框，先用 inspect_page 检查页面布局，再用 crop_target 裁剪目标图像；证据 id 可直接由 open_evidence/write_claims_chunk 使用，不调用 select_evidence。"
        )
    if schema in {"region", "evidence_select", "chunked_claim"}:
        return (
            "坐标规则：所有候选 region 的 bbox 都使用第 1 张 PDF 原始页面图像的像素坐标，原点在左上角，格式为 [x1,y1,x2,y2]；"
            "当前页面没有红框，必须先用 propose_regions 查看候选区域；下一步工具必须服从“当前阶段允许的工具”列表。"
            "如果当前阶段只允许 crop_region，就必须先用 crop_region(region_id) 查看目标区域，不能提前 select_evidence。"
        )
    return "坐标规则：所有 bbox 都使用第 1 张 PDF 页面图像的像素坐标，原点在左上角，格式为 [x1,y1,x2,y2]；如果页面上有红框，crop_image 必须裁剪红框范围。"


def simplify_action(action: Any) -> Any:
    if not isinstance(action, dict):
        return action
    keep = {
        key: action.get(key)
        for key in [
            "action",
            "bbox",
            "region_id",
            "field",
            "evidence_id",
            "scope",
            "top_k",
            "value",
            "reason",
            "status",
        ]
        if key in action
    }
    if "query" in action:
        keep["query"] = truncate_text(str(action.get("query", "")), 120)
    if "anchor" in action:
        keep["anchor"] = action.get("anchor")
    if "evidence_ids" in action:
        keep["evidence_ids"] = action.get("evidence_ids")
    if "claims" in action:
        keep["claims"] = simplify_claim_items(action.get("claims") or [])
    if "abstains" in action:
        keep["abstains"] = simplify_abstain_items(action.get("abstains") or [])
    return keep


def simplify_tool_result(result: Any, config: PromptConfig) -> Any:
    if not isinstance(result, dict):
        return result
    tool = result.get("tool")
    if tool in {"crop_image", "crop_region", "crop_target"}:
        keep = {
            "tool": tool,
            "bbox": result.get("bbox"),
            "crop_path": result.get("crop_path"),
            "bbox_iou": round(float(result.get("bbox_iou", -1)), 4)
            if result.get("bbox_iou") is not None
            else None,
        }
        if "region_id" in result:
            keep["region_id"] = result.get("region_id")
        return keep
    if tool in {"propose_regions", "inspect_page"} and result.get("regions"):
        region_limit = max(10, config.max_evidence_per_result * 2)
        return {
            "tool": tool,
            "page_size": result.get("page_size"),
            "regions": [
                {
                    "region_id": item.get("region_id"),
                    "bbox": item.get("bbox"),
                    "type": item.get("type"),
                    "source": item.get("source"),
                    "score": item.get("score"),
                    "nearby_text": truncate_text(str(item.get("nearby_text", "")), 120) if item.get("nearby_text") else None,
                    "caption_evidence_id": item.get("caption_evidence_id"),
                    "caption_hint": truncate_text(str(item.get("caption_hint", "")), 120) if item.get("caption_hint") else None,
                    "hint": truncate_text(str(item.get("hint", "")), 120) if item.get("hint") else None,
                    "linked_caption_region_id": item.get("linked_caption_region_id"),
                    "linked_caption_text": truncate_text(str(item.get("linked_caption_text", "")), 160)
                    if item.get("linked_caption_text")
                    else None,
                    "linked_caption_position": item.get("linked_caption_position"),
                    "linked_caption_gap_px": item.get("linked_caption_gap_px"),
                    "caption_link_score": item.get("caption_link_score"),
                    "target_caption_match_score": item.get("target_caption_match_score"),
                    "target_caption_match_reason": item.get("target_caption_match_reason"),
                    "target_region_rank": item.get("target_region_rank"),
                    "target_region_sort_score": item.get("target_region_sort_score"),
                }
                for item in (result.get("regions") or [])[:region_limit]
            ],
        }
    if tool == "select_evidence":
        return {
            "tool": "select_evidence",
            "selected_evidence_ids": result.get("selected_evidence_ids") or [],
            "selected_evidence": [
                simplify_evidence(item, config)
                for item in (result.get("selected_evidence") or [])[: config.max_evidence_per_result]
            ],
            "rejected_evidence_ids": result.get("rejected_evidence_ids") or [],
        }
    if tool == "retrieve_evidence":
        return {
            "tool": "retrieve_evidence",
            "scope": result.get("scope"),
            "query": truncate_text(str(result.get("query", "")), 120),
            "anchor": result.get("anchor"),
            "results": [
                simplify_evidence(item, config)
                for item in (result.get("results") or [])[: config.max_evidence_per_result]
            ],
            "hit_evidence_ids": result.get("hit_evidence_ids") or [],
        }
    if tool == "open_evidence":
        simplified = {"tool": "open_evidence", "evidence_id": result.get("evidence_id")}
        if result.get("error"):
            simplified["error"] = result.get("error")
        for key in ["source_file", "page_start", "page_end", "authority_level", "citation_level", "source_quality"]:
            if key in result:
                simplified[key] = result.get(key)
        for key in EVIDENCE_POLICY_KEYS:
            if key in result and result.get(key) is not None:
                simplified[key] = result.get(key)
        if config.field_policy_prompt and result.get("claim_use_hint"):
            simplified["claim_use_hint"] = result.get("claim_use_hint")
        for key in ["display_snippet", "evidence_summary", "text", "raw_chunk_text"]:
            if result.get(key):
                simplified[key] = truncate_text(str(result.get(key)), config.snippet_chars)
                break
        return simplified
    if tool in {"write_claims_chunk", "write_claims_batch"}:
        return {
            "tool": tool,
            "claims": simplify_claim_items(result.get("claims") or []),
            "abstains": simplify_abstain_items(result.get("abstains") or []),
            "claim_state": result.get("claim_state"),
        }
    if tool in {"write_claim", "abstain_claim"}:
        return {
            "tool": tool,
            "claim": result.get("claim"),
            "claim_state": result.get("claim_state"),
        }
    return {key: result.get(key) for key in list(result)[:8]}


def simplify_evidence(item: Any, config: PromptConfig) -> Any:
    if not isinstance(item, dict):
        return item
    snippet = item.get("evidence_summary") or item.get("display_snippet") or item.get("text") or ""
    simplified = {
        "evidence_id": item.get("evidence_id"),
        "source_file": item.get("source_file"),
        "page_start": item.get("page_start"),
        "page_end": item.get("page_end"),
        "authority_level": item.get("authority_level"),
        "citation_level": item.get("citation_level"),
        "score": item.get("score"),
        "snippet": truncate_text(str(snippet), config.snippet_chars),
    }
    for key in EVIDENCE_POLICY_KEYS:
        if key in item and item.get(key) is not None:
            simplified[key] = item.get(key)
    if config.field_policy_prompt and item.get("claim_use_hint"):
        simplified["claim_use_hint"] = item.get("claim_use_hint")
    return simplified


def truncate_text(text: str, max_chars: int) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def simplify_claim_items(items: list[Any]) -> list[Any]:
    simplified: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            simplified.append(item)
            continue
        simplified.append(
            {
                "field": item.get("field"),
                "value": truncate_text(str(item.get("value")), 160) if item.get("value") is not None else None,
                "evidence_ids": item.get("evidence_ids") or [],
                "visual_bbox": item.get("visual_bbox"),
                "confidence": item.get("confidence"),
            }
        )
    return simplified


def simplify_abstain_items(items: list[Any]) -> list[Any]:
    simplified: list[Any] = []
    for item in items:
        if not isinstance(item, dict):
            simplified.append(item)
            continue
        simplified.append(
            {
                "field": item.get("field"),
                "reason": truncate_text(str(item.get("reason")), 160),
            }
        )
    return simplified


def claim_state_from_claims(claims: list[dict[str, Any]]) -> dict[str, Any]:
    target_fields = CLAIM_FIELD_SPEC.split("|")
    by_field = {str(item.get("field")): item for item in claims if isinstance(item, dict) and item.get("field")}
    written_fields = [field for field in target_fields if field in by_field and not by_field[field].get("abstain")]
    abstained_fields = [field for field in target_fields if field in by_field and by_field[field].get("abstain")]
    return {
        "target_fields": target_fields,
        "written_fields": written_fields,
        "abstained_fields": abstained_fields,
        "remaining_fields": [field for field in target_fields if field not in by_field],
        "claim_count": len(written_fields),
        "abstain_count": len(abstained_fields),
    }


def compact_messages(messages: list[dict[str, Any]], max_text_chars: int, head_text_chars: int) -> list[dict[str, Any]]:
    if max_text_chars <= 0:
        return messages
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    item["text"] = compact_text(item["text"], max_text_chars, head_text_chars)
        elif isinstance(content, str) and message.get("role") != "assistant":
            message["content"] = compact_text(content, max_text_chars, head_text_chars)
    return messages


def compact_text(text: str, max_chars: int, head_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head_chars = min(max(512, head_chars), max_chars - 512)
    tail_chars = max_chars - head_chars
    return (
        text[:head_chars]
        + "\n\n[中间过长的历史/证据返回已为 rollout 截断，保留开头任务定义和结尾当前状态。]\n\n"
        + text[-tail_chars:]
    )


def image_size(path: str | Path | None) -> dict[str, int | None]:
    if not path:
        return {"width": None, "height": None}
    key = str(path)
    if key in IMAGE_SIZE_CACHE:
        width, height = IMAGE_SIZE_CACHE[key]
        return {"width": width, "height": height}
    try:
        from PIL import Image

        with Image.open(key) as image:
            size = image.size
        IMAGE_SIZE_CACHE[key] = size
        return {"width": size[0], "height": size[1]}
    except Exception:
        return {"width": None, "height": None}
