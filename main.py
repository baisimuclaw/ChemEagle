import json
import base64
from typing import Any, Dict, Optional

from chemeagle_llm import (
    LLMBackend,
    LLMRequest,
    backend_model,
    backend_scope,
    create_backend,
)
from chemeagle_llm.base import parse_json_content
from chemeagle_vision import (
    VisionBackend,
    create_vision_backend,
    vision_scope,
)
from chemeagle_vision.request_cache import compact_vision_tool_value

def _normalize_agent_args(raw_args: Optional[dict], image_path: str) -> dict:
    if not isinstance(raw_args, dict):
        return {"image_path": image_path}
    normalized = dict(raw_args)
    placeholder_values = {"[img]", "<img>", "[image]", "<image>", "<<<IMAGE>>>", "IMAGE_PATH", "image.png","image_path"}
    if normalized.get("image_path") in placeholder_values or normalized.get("image_path") is None:
        normalized["image_path"] = image_path
    return normalized


def _chemeagle_cloud_impl(
    image_path: str,
    *,
    backend: LLMBackend,
    llm_model: Optional[str] = None,
    use_plan_observer: bool = False,
    use_action_observer: bool = False,
) -> dict:
    """
    Given a chemical reaction image path, extract reaction information
    using GPT models and specialized agents, and return structured reaction data.
    Supports plan observer and action observer. Default set to False to save token and time.

    Args:
        image_path (str): Path to the image file.
        use_plan_observer (bool): Whether to use plan observer to review the agent call plan.
        use_action_observer (bool): Whether to use action observer to check execution results.

    Returns:
        dict: Structured reaction data including reactants, products, and reaction template.
    """
    from get_R_group_sub_agent import (
        get_full_reaction_template,
        get_multi_molecular_full,
        process_reaction_image_with_product_variant_R_group,
        process_reaction_image_with_table_R_group,
    )
    from get_observer import action_observer_agent, plan_observer_agent
    from get_reaction_agent import get_reaction_con
    from get_text_agent import text_extraction_agent
    from chemietoolkit.helper import (
        _parse_planner_output,
        _resolve_ordered_agents,
        fallback_resolve_condition_smiles_in_data,
        fallback_resolve_reactant_product_smiles_in_data,
        fallback_validate_and_fix_smiles_in_dict,
    )

    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)


    agent_specs = [
        {
        'type': 'function',
        'function': {
            'name': 'process_reaction_image_with_product_variant_R_group',
            'description': 'get the reaction data of the reaction diagram and get SMILES strings of every detailed reaction in reaction diagram and the set of product variants, and the original molecular list.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
            {
        'type': 'function',
        'function': {
            'name': 'process_reaction_image_with_table_R_group',
            'description': 'get the reaction data of the reaction diagram and get SMILES strings of every detailed reaction in reaction diagram and the R-group table',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
            {
        'type': 'function',
        'function': {
            'name': 'get_full_reaction_template',
            'description': 'After you carefully check the image, if this is a reaction image that contains only a text-based table and does not involve any R-group replacement, or this is a reaction image does not contain any tables or sets of product variants, then just call this simplified agent.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
            {
        'type': 'function',
        'function': {
            'name': 'get_multi_molecular_full',
            'description': 'After you carefully check the image, if this is a single molecule image or a multiple molecules image, then need to call this molecular recognition agent.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
            },
        {
        'type': 'function',
        'function': {
            'name': 'get_reaction_con',
            'description': 'Extract and normalize the reaction conditions (catalysts, reagents, solvent, temperature, time, atmosphere, etc.) from the graphic. Call this condition interpretation agent when the image contains explicit reaction conditions but no R-group tables or sets of product variants (the R-group agents already interpret conditions internally).',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
        },
        {
        'type': 'function',
        'function': {
            'name': 'text_extraction_agent',
            'description': 'Extract the text from the image.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'image_path': {
                        'type': 'string',
                        'description': 'The path to the reaction image.',
                    },
                },
                'required': ['image_path'],
                'additionalProperties': False,
            },
        },
        },
    ]

    with open('./prompt/prompt_final_simple_version.txt', 'r', encoding='utf-8') as prompt_file:
    #with open('./prompt/prompt_data.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    with open('./prompt/prompt_plan.txt', 'r', encoding='utf-8') as prompt_file:
        planner_user_message = prompt_file.read()

    planner_response = backend.generate(LLMRequest(
        model=backend_model(backend, llm_model or 'gpt-5-mini'),
        messages=[
            {'role': 'system', 'content': "You are a chemical image understanding and extraction planning expert. After checking the image, your ONLY task is to SELECT and CALL the most appropriate agents from the list below to best fit the data extraction of the image."},
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': planner_user_message},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
                ]
            }
        ],
    ))
    
    planner_output = (planner_response.content or "").strip()
    print(f"[D] Planner output: {planner_output}")
    
    agent_list = _parse_planner_output(planner_output)
    print(f"[D] Parsed agents: {agent_list}")
    
    if use_plan_observer:
        observer_output = plan_observer_agent(image_path, agent_list)
        reviewed = observer_output.get("list_of_agents", agent_list)
        reason = observer_output.get("reason", "")
        if isinstance(reviewed, list) and reviewed:
            new_agents = []
            for item in reviewed:
                if isinstance(item, str):
                    new_agents.append(item)
                elif isinstance(item, dict):
                    name = item.get("name") or item.get("tool_name") or ""
                    if name:
                        new_agents.append(name)
            if new_agents:
                agent_list = new_agents
                print(f"[D] Plan observer revised agents: {agent_list}")
                if reason:
                    print(f"[D] Plan observer reason: {reason}")
    

    ordered_agents, has_text_extraction = _resolve_ordered_agents(agent_list)

    AGENT_MAP = {
        'process_reaction_image_with_product_variant_R_group': process_reaction_image_with_product_variant_R_group,
        'process_reaction_image_with_table_R_group': process_reaction_image_with_table_R_group,
        'get_full_reaction_template': get_full_reaction_template,
        'get_multi_molecular_full': get_multi_molecular_full,
        'get_reaction_con': get_reaction_con,
        'text_extraction_agent': text_extraction_agent
    }

    execution_logs = []
    results = []
    main_area_result = None
    observer_notes = []
    failed_agents = []

    def _observe_and_retry(observed_name, observed_result, rerun):
        """Per-agent Action Observer check; at most one re-execution on redo.
        The observer's diagnosis is collected and forwarded to the final
        synthesis step."""
        check = action_observer_agent(
            image_path, [{"name": observed_name, "result": observed_result}])
        if check.get("redo"):
            reason = check.get("reason", "")
            observer_notes.append(f"{observed_name}: {reason}" if reason else observed_name)
            print(f"[D] Action observer requested redo for {observed_name}: {reason}")
            observed_result = rerun()
        return observed_result

    for idx, agent_name in enumerate(ordered_agents):
        print(f"[D] Executing agent {idx + 1}/{len(ordered_agents)}: {agent_name}")
        try:
            agent_result = AGENT_MAP[agent_name](image_path=image_path)
            if use_action_observer:
                agent_result = _observe_and_retry(
                    agent_name, agent_result,
                    lambda: AGENT_MAP[agent_name](image_path=image_path))
        except Exception as exc:
            failed_agents.append(f"{agent_name}: {type(exc).__name__}: {exc}")
            observer_notes.append(f"{agent_name} failed and was skipped ({type(exc).__name__})")
            print(f"[D] Agent {agent_name} failed, continuing without it: {exc!r}")
            continue
        if main_area_result is None:
            main_area_result = agent_result
        execution_logs.append({
            "id": f"agent_call_{idx}",
            "name": agent_name,
            "arguments": {"image_path": image_path},
            "result": agent_result,
        })
        results.append({
            'role': 'tool',
            'content': json.dumps({
                'image_path': image_path,
                agent_name: compact_vision_tool_value(agent_result),
            }),
            'tool_call_id': f"agent_call_{idx}",
        })

    if not results:
        raise RuntimeError(
            "All planned agents failed, nothing left to synthesise: "
            + " | ".join(failed_agents)
        )

    observer_reason = "; ".join(observer_notes)

    text_extraction_result = None
    if has_text_extraction:
        print(f"[D] Executing text_extraction_agent with graphical_input")
        try:
            text_extraction_result = text_extraction_agent(
                image_path=image_path,
                graphical_input=compact_vision_tool_value(main_area_result),
            )
            if use_action_observer and text_extraction_result is not None:
                text_extraction_result = _observe_and_retry(
                    "text_extraction_agent", text_extraction_result,
                    lambda: text_extraction_agent(
                        image_path=image_path,
                        graphical_input=compact_vision_tool_value(main_area_result)))
        except Exception as exc:

            failed_agents.append(f"text_extraction_agent: {type(exc).__name__}: {exc}")
            observer_notes.append(f"text_extraction_agent failed and was skipped ({type(exc).__name__})")
            print(f"[D] Agent text_extraction_agent failed, continuing without it: {exc!r}")
            text_extraction_result = None
        observer_reason = "; ".join(observer_notes)


    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": res['tool_call_id'],
                "type": "function",
                "function": {
                    "name": log["name"],
                    "arguments": json.dumps({"image_path": image_path}),
                },
            }
            for log, res in zip(execution_logs, results)
        ],
    }

    messages_list = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {
                    'type': 'text',
                    'text': prompt
                },
                {
                    'type': 'image_url',
                    'image_url': {
                        'url': f'data:image/png;base64,{base64_image}'
                    }
                }
            ]
        },
        assistant_message,
        *results,
    ]

    if text_extraction_result is not None:
        if isinstance(text_extraction_result, dict) and "annotated_text" in text_extraction_result:
            _text_extraction_for_msg = {"annotated_text": text_extraction_result["annotated_text"]}
        else:
            _text_extraction_for_msg = text_extraction_result
        messages_list.append({
        "role": "user",
        "content": (
            "Additionally, the text_extraction_agent has produced the following "
            "JSON for the prose portion of the same image."
            "```json\n"
            + json.dumps(_text_extraction_for_msg, ensure_ascii=False, indent=2)
            + "\n```"
        ),
    })
    
    if observer_reason:
        messages_list.append({
            'role': 'user',
            'content': f"Note: the previous execution had potential errors: {observer_reason}. Please review the results carefully.",
        })

    response = backend.generate(LLMRequest(
        model=backend_model(backend, llm_model or 'gpt-4o'),
        messages=messages_list,
        json_mode=True,
        temperature=0,
    ))

    gpt_output = parse_json_content(response)
    gpt_output = fallback_validate_and_fix_smiles_in_dict(gpt_output)
    gpt_output = fallback_resolve_condition_smiles_in_data(gpt_output)
    gpt_output = fallback_resolve_reactant_product_smiles_in_data(gpt_output)    


    
    if text_extraction_result is not None:
        if isinstance(text_extraction_result, dict) and "annotated_text" in text_extraction_result:
            gpt_output["text_extraction"] = [{"annotated_text": text_extraction_result["annotated_text"]}]
        else:
            gpt_output["text_extraction"] = [text_extraction_result]
        
    print(gpt_output)
    return gpt_output



def ChemEagle(
    image_path: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    backend: Optional[LLMBackend] = None,
    vision_provider: Optional[str] = None,
    vision_backend: Optional[VisionBackend] = None,
    vision_options: Optional[Dict[str, Any]] = None,
    use_plan_observer: bool = False,
    use_action_observer: bool = False,
) -> dict:
    """Extract structured chemical data using the selected LLM backend.

    ``provider`` selects the language model. ``vision_provider`` independently
    selects local, SSH, or Slurm-over-SSH execution for the chemical vision
    models. Supplying already-created backends reuses both persistent workers.
    """
    selected = backend or create_backend(
        provider, model=model, base_url=base_url, api_key=api_key
    )
    selected_vision = vision_backend or create_vision_backend(
        vision_provider, **(vision_options or {})
    )
    owns_backend = backend is None
    owns_vision_backend = vision_backend is None
    try:
        with backend_scope(selected), vision_scope(selected_vision):
            return _chemeagle_cloud_impl(
                image_path,
                backend=selected,
                llm_model=model,
                use_plan_observer=use_plan_observer,
                use_action_observer=use_action_observer,
            )
    finally:
        if owns_backend:
            selected.close()
        if owns_vision_backend:
            selected_vision.close()


def ChemEagle_OS(
    image_path: str,
    *,
    model_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    vision_provider: Optional[str] = None,
    vision_backend: Optional[VisionBackend] = None,
    vision_options: Optional[Dict[str, Any]] = None,
    use_plan_observer: bool = False,
    use_action_observer: bool = False,
) -> dict:
    """Backward-compatible wrapper for ``ChemEagle(provider="local")``."""
    return ChemEagle(
        image_path,
        provider="local",
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        vision_provider=vision_provider,
        vision_backend=vision_backend,
        vision_options=vision_options,
        use_plan_observer=use_plan_observer,
        use_action_observer=use_action_observer,
    )
