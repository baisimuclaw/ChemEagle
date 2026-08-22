import base64
import json
import logging
import os
from typing import Any, List, Optional

from chemeagle_llm import LLMRequest, backend_model, get_active_backend
from chemeagle_llm.base import parse_json_content


logger = logging.getLogger(__name__)

PLAN_PROMPT_TEMPLATE = """System Message: 
You are a plan observer. Given the graphic and the current list of agent calls (plan), decide whether the plan is sufficient.

User Message: 
Given the graphic and the current list of agent calls (plan), please recheck the component of the graphic and decide whether the corresponding agent is called and the plan is sufficient.
Valid agents:
1. Reaction template parsing agent
Parses the reaction scheme to identify reactants, products, and label mappings, and outputs a structured reaction template.

2. Molecular recognition agent
Detects other molecules in the graphic except in the reaction template, recognizes their structures, and returns normalized representations (e.g., SMILES, labels, positions).

3. Structure-based R-group substitution agent
Uses structure panels / variant images to extract R-group values and generate enumerated products from a core scaffold based on structural information.

4. Text-based R-group substitution agent
Reads R-group tables and enumerates products or substituents on top of a given core scaffold using text information.

5. Condition interpretation agent
Extracts and normalizes reaction conditions (catalysts, reagents, solvent, temperature, time, atmosphere, etc.) from the graphic.

6. Text extraction agent
Performs chmical NER and text-based reaction extraction on the text description.

If the plan is acceptable, return the original plan as-is.
If adjustments are required, provide the improved list of agents and briefly explain the changes.

Key distinction between Structure-based and Text-based R-group agents:
The deciding factor is what each TABLE CELL contains:
- COMPLETE product/substrate molecules (full structures) → Structure-based R-group substitution agent
- R-group FRAGMENTS or substituent values (text like "Me"/"Ph", or partial structures with wavy bonds showing only the R-group portion) → Text-based R-group substitution agent
Even if every cell is a drawn structure, if those structures are R-group FRAGMENTS (not complete products), use the Text-based agent.

Always respond in valid JSON with the structure:
{{
  "list_of_agents": [...],
  "redo": true/false,
  "reason": "If changed is true, give an explanation; otherwise leave blank."
}}

Current plan (JSON):
{plan_json}
"""

ACTION_PROMPT_TEMPLATE = """System Message: 
You are an action observer. Your task is to observe the graphic and the current agent output, decide whether the agent must be rerun.

User Message: 
By observing the image and the current agent output, decide whether the agent must be rerun.
The main focus is on whether the SMILES is reasonable and effective. Is the condition or text classification correct?
If the outcome is acceptable, return redo=false.
If issues are found or corrections are needed, return redo=true with a short explanation.

Always respond in valid JSON with the structure:
{{
  "redo": true/false,
  "reason": "Provide the reasons when redo is true; otherwise leave blank.",
}}

Current agent_result (JSON):
{result_json}
"""


def _encode_image(image_path: str) -> str | None:
    if not image_path or not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def plan_observer_agent(image_path: str, tool_calls: List[Any]) -> dict:
    """Returns {"list_of_agents": list, "redo": bool, "reason": str}."""
    default = {"list_of_agents": tool_calls, "redo": False, "reason": ""}
    base64_image = _encode_image(image_path)
    plan_json = json.dumps(tool_calls or [], ensure_ascii=False, indent=2)
    prompt = PLAN_PROMPT_TEMPLATE.format(plan_json=plan_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        backend = get_active_backend()
        response = backend.generate(LLMRequest(
            model=backend_model(backend, "gpt-5-mini"),
            json_mode=True,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
        ))
        parsed = parse_json_content(response)
        return {
            "list_of_agents": parsed.get("list_of_agents", parsed.get("plan", tool_calls)),
            "redo": bool(parsed.get("redo", False)),
            "reason": parsed.get("reason", ""),
        }
    except Exception as exc:
        logger.warning("Plan observer degraded to the original plan: %s", exc)
        return default


def action_observer_agent(image_path: str, tool_result: Any) -> dict:
    """Returns {"redo": bool, "reason": str, "list_of_agents": list}."""
    default = {"redo": False, "reason": "", "list_of_agents": []}
    base64_image = _encode_image(image_path)
    result_json = json.dumps(tool_result, ensure_ascii=False, indent=2)
    prompt = ACTION_PROMPT_TEMPLATE.format(result_json=result_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        backend = get_active_backend()
        response = backend.generate(LLMRequest(
            model=backend_model(backend, "gpt-5-mini"),
            json_mode=True,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
        ))
        parsed = parse_json_content(response)
        return {
            "redo": bool(parsed.get("redo", False)),
            "reason": parsed.get("reason", ""),
            "list_of_agents": parsed.get("list_of_agents", []),
        }
    except Exception as exc:
        logger.warning("Action observer degraded to no redo: %s", exc)
        return default


def plan_observer_agent_OS(
    image_path: str,
    tool_calls: List[Any],
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    OS version of plan_observer_agent, using a local/self-hosted model compatible with the OpenAI Chat Completions protocol.

    Returns:
        dict: {"list_of_agents": list, "redo": bool, "reason": str}
    """
    default = {"list_of_agents": tool_calls, "redo": False, "reason": ""}
    backend = get_active_backend(
        provider="local", model=model_name, base_url=base_url, api_key=api_key
    )

    base64_image = _encode_image(image_path)
    plan_json = json.dumps(tool_calls or [], ensure_ascii=False, indent=2)
    prompt = PLAN_PROMPT_TEMPLATE.format(plan_json=plan_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        response = backend.generate(LLMRequest(
            model=backend_model(backend, model_name),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            json_mode=True,
        ))
        parsed = parse_json_content(response)
        
        return {
            "list_of_agents": parsed.get("list_of_agents", parsed.get("plan", tool_calls)),
            "redo": bool(parsed.get("redo", False)),
            "reason": parsed.get("reason", ""),
        }
    except Exception as e:
        logger.warning("Local plan observer degraded to the original plan: %s", e)
        return default


def action_observer_agent_OS(
    image_path: str,
    tool_result: Any,
    *,
    model_name: str = "/models/Qwen3-VL-32B-Instruct",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    """
    OS version of action_observer_agent, using a local/self-hosted model compatible with the OpenAI Chat Completions protocol.

    Returns:
        dict: {"redo": bool, "reason": str, "list_of_agents": list}
    """
    default = {"redo": False, "reason": "", "list_of_agents": []}
    backend = get_active_backend(
        provider="local", model=model_name, base_url=base_url, api_key=api_key
    )

    base64_image = _encode_image(image_path)
    result_json = json.dumps(tool_result, ensure_ascii=False, indent=2)
    prompt = ACTION_PROMPT_TEMPLATE.format(result_json=result_json)

    user_content = [{"type": "text", "text": prompt}]
    if base64_image:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_image}"},
            }
        )

    try:
        response = backend.generate(LLMRequest(
            model=backend_model(backend, model_name),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            json_mode=True,
        ))
        parsed = parse_json_content(response)
        
        return {
            "redo": bool(parsed.get("redo", False)),
            "reason": parsed.get("reason", ""),
            "list_of_agents": parsed.get("list_of_agents", []),
        }
    except Exception as e:
        logger.warning("Local action observer degraded to no redo: %s", e)
        return default
