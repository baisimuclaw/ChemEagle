import sys
import torch
import json
from chemietoolkit import ChemIEToolkit
import cv2
from PIL import Image
import json
import sys
import torch
from rxnim import RxnIM
import json
from molnextr.chemistry import _convert_graph_to_smiles
import base64
import numpy as np
from chemietoolkit import utils
from PIL import Image
import os
from typing import Optional
from chemietoolkit.helper import _patch_to_reaction
from chemeagle_llm import (
    LLMRequest,
    backend_model,
    bind_image_tools,
    get_active_backend,
    parse_json_content,
)



ckpt_path = "./rxn.ckpt"
model1 = RxnIM(ckpt_path, device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
model = ChemIEToolkit(device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def _run_image_tool_agent(
    backend,
    image_path,
    messages,
    tools,
    model_name,
    handlers,
    *,
    extra=None,
):
    response = backend.run_tool_loop(
        LLMRequest(
            model=backend_model(backend, model_name),
            messages=messages,
            json_mode=True,
            temperature=0,
            tool_choice="auto",
            extra=extra or {},
        ),
        tools,
        bind_image_tools(image_path, handlers),
    )
    return parse_json_content(response)


def get_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image_file = image_path
    image = Image.open(image_file)

    image_file = image_path
    raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
    #print(f'raw_prediction:{raw_prediction}')

    # Ensure raw_prediction is treated as a list directly
    structured_output = {}
    for section_key in ['reactants', 'conditions', 'products']:
        if section_key in raw_prediction[0]:
            structured_output[section_key] = []
            for item in raw_prediction[0][section_key]:
                if section_key in ['reactants', 'products']:
                    # Extract smiles and bbox for molecules
                    structured_output[section_key].append({
                        "smiles": item.get("smiles", ""),
                        "bbox": item.get("bbox", []),
                        "symbols": item.get("symbols", [])  
                    })
                elif section_key == 'conditions':
                    # Extract smiles, text, and bbox for conditions
                    condition_data = {"bbox": item.get("bbox", [])}
                    if "smiles" in item:
                        condition_data["smiles"] = item.get("smiles", "")
                        condition_data["symbols"] = item.get("symbols", [])
                    if "text" in item:
                        condition_data["text"] = item.get("text", [])
                    structured_output[section_key].append(condition_data)
    #print(f'structured_output:{structured_output}')

    return structured_output



def get_full_reaction(image_path: str) -> dict:
    '''
    Returns a structured dictionary of reactions extracted from the image,
    including reactants, conditions, and products, with their smiles, text, and bbox.
    '''
    image_file = image_path
    raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
    for reaction in raw_prediction:
        for section in ("reactants", "products", "conditions"):
            for entry in reaction.get(section, []):
                # 1) Keep coords to three decimal places
                coords = entry.get("coords")
                if isinstance(coords, list):
                    entry["coords"] = [
                        [round(val, 3) for val in point]
                        for point in coords
                    ]
                # 2) Remove unnecessary fields
                for key in ("molfile", "atoms", "bonds"):
                    entry.pop(key, None)

    raw_prediction =json.dumps(raw_prediction)
    return raw_prediction



def get_reaction_withatoms(image_path: str) -> dict:
    """
    Input a chemical reaction image path, use GPT and OpenChemIE to extract reaction information, and return organized reaction data.

    Args:
        image_path (str): image file path.

    Returns:
        dict: organized reaction data, including reactants, products, and reaction templates.
    """
    backend = get_active_backend()

    # Load image and encode as Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    # GPT tool-calling configuration
    tools = [
        {
        'type': 'function',
        'function': {
            'name': 'get_reaction',
            'description': 'Get a list of reactions from a reaction image. A reaction contains data of the reactants, conditions, and products.',
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

    # Message content provided to GPT
    with open('./prompt/prompt_getreaction.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
            ]
        }
    ]

    gpt_output = _run_image_tool_agent(
        backend,
        image_path,
        messages,
        tools,
        "gpt-4o",
        {"get_reaction": get_reaction},
    )
    #print(f"gpt_output1:{gpt_output}")

    
    def get_reaction_full(image_path: str) -> dict:
        '''
        Returns a structured dictionary of reactions extracted from the image,
        including reactants, conditions, and products, with their smiles, text, and bbox.
        '''
        image_file = image_path
        raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
        return raw_prediction
    
    input2 = get_reaction_full(image_path)



    def update_input_with_symbols(input1, input2, conversion_function):
        symbol_mapping = {}
        for key in ['reactants', 'products']:
            for item in input1.get(key, []):
                bbox = tuple(item['bbox'])  # Use bbox as a unique identifier
                symbol_mapping[bbox] = item['symbols']

        for key in ['reactants', 'products']:
            for item in input2.get(key, []):
                bbox = tuple(item['bbox'])  # Get bbox as matching key

                # If bbox exists in input1 mapping, update symbols
                if bbox in symbol_mapping:
                    updated_symbols = symbol_mapping[bbox]
                    item['symbols'] = updated_symbols
                    
                    # Update atom_symbol in atoms
                    if 'atoms' in item:
                        atoms = item['atoms']
                        if len(atoms) != len(updated_symbols):
                            print(f"Warning: Mismatched symbols and atoms in bbox {bbox}")
                        else:
                            for atom, symbol in zip(atoms, updated_symbols):
                                atom['atom_symbol'] = symbol
                    
                    # If coords and edges exist, call conversion function to generate new smiles and molfile
                    if 'coords' in item and 'edges' in item:
                        coords = item['coords']
                        edges = item['edges']
                        new_smiles, new_molfile, _ = conversion_function(coords, updated_symbols, edges)
                        
                        # Replace old smiles and molfile
                        item['smiles'] = new_smiles
                        item['molfile'] = new_molfile

        return input2
    
    updated_data = [update_input_with_symbols(gpt_output, input2[0], _convert_graph_to_smiles)]

    return updated_data

 


def get_reaction_withatoms_correctR(image_path: str) -> dict:
    """
    Input a chemical reaction image path, use GPT and OpenChemIE to extract reaction information, and return organized reaction data.

    Args:
        image_path (str): image file path.

    Returns:
        dict: organized reaction data, including reactants, products, and reaction templates.
    """
    backend = get_active_backend()

    # Load image and encode as Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    # GPT tool-calling configuration
    tools = [
        {
        'type': 'function', 
        'function': {
            'name': 'get_reaction',
            'description': 'Get a list of reactions from a reaction image. A reaction contains data of the reactants, conditions, and products.',
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

    # Message content provided to GPT
    with open('./prompt/prompt_Rxn_Tem.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
            ]
        }
    ]

    gpt_output = _run_image_tool_agent(
        backend,
        image_path,
        messages,
        tools,
        "gpt-5-mini",
        {"get_reaction": get_reaction},
    )
    print(f"gpt_output_rxn:{gpt_output}")

    
    def get_reaction_full(image_path: str) -> dict:
        '''
        Returns a structured dictionary of reactions extracted from the image,
        including reactants, conditions, and products, with their smiles, text, and bbox.
        '''

        image_file = image_path
        raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
        return raw_prediction
    
    input2 = get_reaction_full(image_path)



    def update_input_with_symbols(input1, input2, conversion_function):
        symbol_mapping = {}
        for key in ['reactants', 'conditions', 'products']:
            for item in input1.get(key, []):
                # Only handle items with symbols and bbox fields (conditions may only have text without symbols)
                if 'symbols' in item and 'bbox' in item:
                    bbox = tuple(item['bbox'])  # Use bbox as a unique identifier
                    symbol_mapping[bbox] = item['symbols']

        for key in ['reactants', 'conditions', 'products']:
            for item in input2.get(key, []):
                if 'bbox' not in item:
                    continue
                bbox = tuple(item['bbox'])  # Get bbox as matching key

                # If bbox exists in input1 mapping, update symbols
                if bbox in symbol_mapping:
                    updated_symbols = symbol_mapping[bbox]
                    item['symbols'] = updated_symbols
                    
                    # Update atom_symbol in atoms
                    if 'atoms' in item:
                        atoms = item['atoms']
                        if len(atoms) != len(updated_symbols):
                            print(f"Warning: Mismatched symbols and atoms in bbox {bbox}")
                        else:
                            for atom, symbol in zip(atoms, updated_symbols):
                                atom['atom_symbol'] = symbol
                    
                    # If coords and edges exist, call conversion function to generate new smiles and molfile
                    if 'coords' in item and 'edges' in item:
                        coords = item['coords']
                        edges = item['edges']
                        new_smiles, new_molfile, _ = conversion_function(coords, updated_symbols, edges)
                        
                        # Replace old smiles and molfile
                        item['smiles'] = new_smiles
                        item['molfile'] = new_molfile

        return input2
    
    updated_data = [update_input_with_symbols(gpt_output, input2[0], _convert_graph_to_smiles)]
    updated_data = _patch_to_reaction(updated_data)
    print(f"rxn_agent_output:{updated_data}")

    return updated_data


def get_reaction_withatoms_correctR_OS(
    image_path: str,
    *,
    model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
 

    backend = get_active_backend(
        provider="local",
        model=model_name,
        base_url=base_url,
        api_key=api_key,
    )

    # Load image and encode as Base64
    def encode_image(image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    # GPT tool-calling configuration
    tools = [
        {
            'type': 'function', 
            'function': {
                'name': 'get_reaction',
                'description': 'Get a list of reactions from a reaction image. A reaction contains data of the reactants, conditions, and products.',
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

    # Message content provided to GPT
    with open('./prompt/prompt_Rxn_Tem.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}}
            ]
        }
    ]

    gpt_output = _run_image_tool_agent(
        backend,
        image_path,
        messages,
        tools,
        model_name,
        {"get_reaction": get_reaction},
        extra={"extra_body": _get_extra_body(model_name)},
    )
    
    print(f"gpt_output_rxn:{gpt_output}")

    def get_reaction_full(image_path: str) -> dict:
        '''
        Returns a structured dictionary of reactions extracted from the image,
        including reactants, conditions, and products, with their smiles, text, and bbox.
        '''

        image_file = image_path
        raw_prediction = model1.predict_image_file(image_file, molnextr=True, ocr=True)
        return raw_prediction
    
    input2 = get_reaction_full(image_path)

    def update_input_with_symbols(input1, input2, conversion_function):
        symbol_mapping = {}
        for key in ['reactants', 'conditions', 'products']:
            for item in input1.get(key, []):
                # Only handle items with symbols and bbox fields (conditions may only have text without symbols)
                if 'symbols' in item and 'bbox' in item:
                    bbox = tuple(item['bbox'])  # Use bbox as a unique identifier
                    symbol_mapping[bbox] = item['symbols']

        for key in ['reactants', 'conditions', 'products']:
            for item in input2.get(key, []):
                if 'bbox' not in item:
                    continue
                bbox = tuple(item['bbox'])  # Get bbox as matching key

                # If bbox exists in input1 mapping, update symbols
                if bbox in symbol_mapping:
                    updated_symbols = symbol_mapping[bbox]
                    item['symbols'] = updated_symbols
                    
                    # Update atom_symbol in atoms
                    if 'atoms' in item:
                        atoms = item['atoms']
                        if len(atoms) != len(updated_symbols):
                            print(f"Warning: Mismatched symbols and atoms in bbox {bbox}")
                        else:
                            for atom, symbol in zip(atoms, updated_symbols):
                                atom['atom_symbol'] = symbol
                    
                    # If coords and edges exist, call conversion function to generate new smiles and molfile
                    if 'coords' in item and 'edges' in item:
                        coords = item['coords']
                        edges = item['edges']
                        new_smiles, new_molfile, _ = conversion_function(coords, updated_symbols, edges)
                        
                        # Replace old smiles and molfile
                        item['smiles'] = new_smiles
                        item['molfile'] = new_molfile

        return input2
    
    updated_data = [update_input_with_symbols(gpt_output, input2[0], _convert_graph_to_smiles)]
    updated_data = _patch_to_reaction(updated_data)
    print(f"rxn_agent_output:{updated_data}")

    return updated_data

def _get_extra_body(model_name: str) -> dict:
    if "Qwen3.5" in model_name or "qwen3.5" in model_name:
        return {"chat_template_kwargs": {"enable_thinking": False}, "repetition_penalty": 1.05}
    return {}

def _tesseract_ocr_image(image_path: str) -> str:
    import pytesseract
    img = Image.open(image_path)
    raw_text = pytesseract.image_to_string(img)
    return raw_text


def get_reaction_c(image_path: str) -> dict:
    raw_prediction = model1.predict_image_file(image_path, molnextr=True, ocr=True)
    conditions_per_reaction = []
    for reaction in raw_prediction:
        conds = reaction.get('conditions', [])
        cleaned = []
        for item in conds:
            cleaned.append({
                'text': item.get('text', ''),
                'category': item.get('category', ''),
                'bbox': item.get('bbox', []),
            })
        conditions_per_reaction.append(cleaned)
    return {'conditions': conditions_per_reaction}


def get_reaction_con(image_path: str) -> dict:
    backend = get_active_backend()

    def encode_image(p: str):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'TesseractOCR',
                'description': 'Run Tesseract OCR on the reaction image and return the extracted raw text (including all condition texts).',
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
                'name': 'get_reaction_c',
                'description': 'RxnConInterpreter: extract and initially classify reaction condition texts (reagent/solvent/temperature/yield/etc.) from the reaction image.',
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

    with open('./prompt/prompt_con.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}},
            ],
        },
    ]

    gpt_output = _run_image_tool_agent(
        backend,
        image_path,
        messages,
        tools,
        "gpt-5-mini",
        {
            "TesseractOCR": _tesseract_ocr_image,
            "get_reaction_c": get_reaction_c,
        },
    )
    print(f"gpt_output_con:{gpt_output}")
    return gpt_output


def get_reaction_con_OS(
    image_path: str,
    *,
    model_name: str = "Qwen/Qwen3-VL-32B-Instruct",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> dict:
    backend = get_active_backend(
        provider="local",
        model=model_name,
        base_url=base_url,
        api_key=api_key,
    )

    def encode_image(p: str):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    base64_image = encode_image(image_path)

    tools = [
        {
            'type': 'function',
            'function': {
                'name': 'TesseractOCR',
                'description': 'Run Tesseract OCR on the reaction image and return the extracted raw text (including all condition texts).',
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
                'name': 'get_reaction_c',
                'description': 'RxnConInterpreter: extract and initially classify reaction condition texts (reagent/solvent/temperature/yield/etc.) from the reaction image.',
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

    with open('./prompt/prompt_con.txt', 'r', encoding='utf-8') as prompt_file:
        prompt = prompt_file.read()

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': prompt},
                {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}},
            ],
        },
    ]

    gpt_output = _run_image_tool_agent(
        backend,
        image_path,
        messages,
        tools,
        model_name,
        {
            "TesseractOCR": _tesseract_ocr_image,
            "get_reaction_c": get_reaction_c,
        },
        extra={"extra_body": _get_extra_body(model_name)},
    )

    print(f"gpt_output_con:{gpt_output}")
    return gpt_output
