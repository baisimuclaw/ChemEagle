from PIL import Image
import pytesseract
from typing import Optional
from chemeagle_llm import (
    LLMRequest,
    backend_model,
    bind_image_tools,
    get_active_backend,
    parse_json_content,
)
import json
import base64
import os
import shutil
import re
import sys
from functools import lru_cache
from chemeagle_vision.proxies import (
    vision_chemner as model2,
    vision_chemrxnextractor as rxn_extractor,
)


# Configure Tesseract OCR path (Windows)
def configure_tesseract():
    """Automatically detect and configure the Tesseract OCR executable path"""
    # If already configured, return directly
    if hasattr(pytesseract.pytesseract, 'tesseract_cmd') and pytesseract.pytesseract.tesseract_cmd:
        if os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            return
    
    # Common Windows installation paths (including custom paths under the project directory)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        # Isolated conda/virtual environment used by the orchestrator.
        os.path.join(os.path.dirname(sys.executable), "tesseract"),
        # Custom path under the project directory
        os.path.join(script_dir, "Tesseract-OCR", "tesseract.exe"),
        os.path.join(os.path.dirname(script_dir), "Tesseract-OCR", "tesseract.exe"),
        # Standard installation path
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
    ]
    
    # First try to find it in PATH
    try:
        tesseract_cmd = shutil.which("tesseract")
        if tesseract_cmd and os.path.exists(tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            print(f"✓ Found Tesseract in PATH: {tesseract_cmd}")
            return
    except Exception:
        pass
    
    # If not found in PATH, try common paths
    for path in possible_paths:
        # Normalize path
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            pytesseract.pytesseract.tesseract_cmd = normalized_path
            print(f"✓ Found Tesseract: {normalized_path}")
            return
    
    # If still not found, prompt the user
    print("⚠️  Warning: Tesseract OCR executable not found")
    print("Paths tried:")
    for path in possible_paths:
        normalized_path = os.path.normpath(path)
        exists = "✓" if os.path.exists(normalized_path) else "✗"
        print(f"  {exists} {normalized_path}")
    print("\nPlease do one of the following:")
    print("1. Make sure Tesseract OCR is installed correctly")
    print("2. Or set the path manually:")
    print("   pytesseract.pytesseract.tesseract_cmd = r'C:\\path\\to\\tesseract.exe'")
    raise FileNotFoundError(
        "Tesseract OCR is not installed or not in PATH."
        "Please visit https://github.com/UB-Mannheim/tesseract/wiki for installation."
    )

def merge_sentences(sentences):
    """
    Merge a list of sentence fragments into a coherent paragraph string.
    """
    # Trim whitespace around each fragment and remove empty strings
    cleaned = [s.strip() for s in sentences if s.strip()]
    # Join with spaces to reconstruct a full paragraph
    paragraph = [" ".join(cleaned)]
    return paragraph


def split_text_into_sentences(text: str) -> list:
    """
    Split text into sentences to avoid issues caused by overly long text.
    Use simple punctuation-based splitting while preserving sentence boundaries.
    """
    # Split by periods, question marks, and exclamation marks, while keeping punctuation
    sentences = re.split(r'([.!?]+)', text)
    # Merge punctuation with preceding text
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = (sentences[i] + sentences[i + 1]).strip()
        else:
            sentence = sentences[i].strip()
        if sentence:
            result.append(sentence)
    
    # If no sentence boundary is found, try splitting by newlines
    if not result:
        result = [line.strip() for line in text.splitlines() if line.strip()]
    
    # If still not found, return the whole text (with length limit)
    if not result:
        # Limit single sentence length to avoid exceeding model limits
        max_length = 500  # character limit
        if len(text) > max_length:
            # Split by spaces into smaller chunks
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_length + word_length > max_length and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word)
                else:
                    current_chunk.append(word)
                    current_length += word_length
            
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            result = chunks
        else:
            result = [text]
    
    return result


def filter_prose_sentences(text: str) -> list:
    """Keep sentence-like prose and discard scheme labels/conditions.

    ChemRxnExtractor is trained for natural-language reaction descriptions, not
    OCR fragments such as ``DBU (1.1 equiv)`` or table rows. Feeding an entire
    scheme's labels to the large text models is both slow and uninformative.
    """
    prose = []
    for sentence in split_text_into_sentences(text):
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
        looks_like_scheme_data = re.search(
            r"(?:=|%|[\[\]{}~+]|\bequiv\b|\bE/Z\b|\bF/Z\b)",
            sentence,
            re.IGNORECASE,
        )
        if (
            len(words) >= 5
            and re.search(r"[.!?](?:\s|$)", sentence)
            and not looks_like_scheme_data
        ):
            prose.append(sentence)
    return prose


def _text_model_sentences(text: str) -> list:
    """Select text-model input; filtering is opt-in to preserve upstream input."""
    enabled = os.environ.get("CHEMEAGLE_TEXT_PROSE_FILTER", "0").strip().lower()
    if enabled in {"0", "false", "no", "off"}:
        return split_text_into_sentences(text)
    return filter_prose_sentences(text)


@lru_cache(maxsize=128)
def _ocr_image_text(image_path: str) -> str:
    configure_tesseract()
    with Image.open(image_path) as img:
        return pytesseract.image_to_string(img)


def extract_reactions_from_text_in_image(image_path: str) -> dict:
    """
    Extract text from a chemical reaction image and identify reactions.

    Arguments:
      image_path: image file path

    Returns:
      {
        'raw_text': full text extracted by OCR (str),
        'paragraph': merged paragraph text (str),
        'reactions': reaction list output by RxnExtractor (list)
      }
    """
    # 1. OCR text extraction
    raw_text = _ocr_image_text(os.path.abspath(image_path))

    # 2. Merge multi-line text into a single paragraph
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)

    # 3. Split text into sentences to avoid length issues
    sentences = _text_model_sentences(paragraph)
    if not sentences:
        print("[Text extraction] No sentence-like prose; skipping ChemRxnExtractor")
        return []
    
    # 4. Extract reactions for each sentence on the active vision worker.
    all_reactions = []
    try:
        reactions = rxn_extractor.get_reactions(sentences)
        all_reactions = reactions
    except AssertionError as e:
        # If it still fails, try processing sentence by sentence
        print(f"Warning: batch processing failed, trying sentence-by-sentence processing: {e}")
        all_reactions = []
        for sent in sentences:
            try:
                sent_reactions = rxn_extractor.get_reactions([sent])
                all_reactions.extend(sent_reactions)
            except Exception as sent_e:
                print(f"Warning: skipping sentence (processing failed): {sent[:50]}... Error: {sent_e}")
                continue

    return all_reactions 

def NER_from_text_in_image(image_path: str) -> dict:
    # 1. OCR text extraction
    raw_text = _ocr_image_text(os.path.abspath(image_path))

    # 2. Merge multi-line text into a single paragraph
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    paragraph = " ".join(lines)
    prose = _text_model_sentences(paragraph)
    if not prose:
        print("[Text extraction] No sentence-like prose; skipping ChemNER")
        return []

    # 3. Extract named entities on the active vision worker.
    predictions = model2.predict_strings([" ".join(prose)])

    return predictions 




def text_extraction_agent(image_path: str, graphical_input: Optional[dict] = None) -> dict:
    """
    Agent that calls two tools:
      1) extract_reactions_from_text_in_image
      2) NER_from_text_in_image
    to perform OCR, reaction extraction, and chemical NER on a single image.
    Returns a merged JSON result.
    """
    backend = get_active_backend()

    # Encode image as Base64
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    # Define tools for the agent
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_reactions_from_text_in_image",
                "description": "OCR image and extract chemical reactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "NER_from_text_in_image",
                "description": "OCR image and perform chemical named entity recognition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"]
                }
            }
        }
    ]

    # Prompt instructing to call both tools
    prompt = """
In this input reaction graphic, there is a chemical reaction scheme template with multiple product/fragment molecular sub-images or tables, conditions, and a text description underneath (or you will receive direct text input).
Your task is to:

1. **Entity Extraction**: First call the `MolNER`("NER_from_text_in_image") tool to extract chemical identities (names, formulas, abbreviations) from the text description.
2. **Reaction Extraction**: Call the `RxnConInterpreter`("extract_reactions_from_text_in_image") tool to identify reaction types and relationships in the text description.
3. **Correction & Alignment**: 
   - Check the tool results and correct any errors.
   - Align the identified text entities with the graphical output (e.g., mapping "precatalyst G32" in text to the molecule G32 in the image).
4. **Output**: Your final output must be in JSON format. Use specific tags like `[reactant]`, `[reagent]`, `[ABBREVIATION]`, `[SMILES:...]` to annotate the original text.

### Example 1
**User Input:**
Text Input: "A visible-light-driven decarboxylative Giese addition of redox-active N-hydroxyphthalimide esters and acrylonitriles was reported by MacMillan and colleagues. Alpha-amino carboxylic acids were utilized as precursors of N-hydroxyphthalimide esters. A highly active PC photocatalyst G32 containing a bulky ligand."

**Assistant Response:**
Here is my step-by-step analysis:

**Step 1: Entity Extraction (MolNER)**
* **Observation:**
    * "N-hydroxyphthalimide esters" -> Chemical (Plural/Class)
    * "acrylonitriles" -> Chemical (Systematic Name)
    * "Alpha-amino carboxylic acids" -> Chemical (Reactant precursor)
    * "PC" -> Abbreviation
    * "G32" -> Chemical (Identifier/Reagent)
* **Action:** Tag these entities with their specific roles and types.

**Step 2: Reaction & Structure Interpretation (RxnConInterpreter)**
* **Observation:**
    * "G32" is identified as a specific precatalyst.
    * **Action:** Retrieve/Associate the SMILES for G32 from the tool/image context: `CCOC(C1=C(OC(C[C@@H]1C2=CC=CC=C2)=O)C)=O,CC1=CC(C)=CC(C)=C1[N+]2=CN3C(CO[C@@H]4[C@H]3C5=C(C=CC=C5)C4)=N2.F[B-](F)(F)F`.
    * "N-hydroxyphthalimide esters" and "acrylonitriles" are participating reactants in a decarboxylative Giese addition.

**Step 3: Text Alignment & Annotation**
* **Thought:** I need to inject the extracted tags and SMILES directly into the original text string to create the final description.
* **Action:** Construct the annotated string.

**Final JSON Output:**
```json
{
  "annotated_text": "A visible-light-driven decarboxylative Giese addition of redox-active N-hydroxyphthalimide esters [reactant][MULTIPLE] and acrylonitriles [reactant][SYSTEMATIC] was reported by MacMillan and colleagues. Alpha-amino carboxylic acids [reactant] were utilized as precursors of N-hydroxyphthalimide esters. A highly active PC [ABBREVIATION] photocatalyst Ir-44 [reagent][IDENTIFIERS][SMILES:CCOC(C1=C(OC(C[C@@H]1C2=CC=CC=C2)=O)C)=O,CC1=CC(C)=CC(C)=C1[N+]2=CN3C(CO[C@@H]4[C@H]3C5=C(C=CC=C5)C4)=N2.F[B-](F)(F)F] containing a bulky ligand."
}
```

"""

    if graphical_input:
        prompt += f"\n\nGraphical extraction results from the reaction image (use this for alignment with text entities):\n{json.dumps(graphical_input, ensure_ascii=False, indent=2)}"

    messages = [
        {"role": "system", "content": "You are the Text Extraction Agent. Your task is to extract text descriptions from chemical reaction images (or process direct text input), identify chemical entities and reactions within that text, and output a structured annotation."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }
    ]

    response = backend.run_tool_loop(
        LLMRequest(
            model=backend_model(backend, "gpt-5-mini"),
            messages=messages,
            json_mode=True,
            tool_choice="auto",
        ),
        tools,
        bind_image_tools(
            image_path,
            {
                "extract_reactions_from_text_in_image": extract_reactions_from_text_in_image,
                "NER_from_text_in_image": NER_from_text_in_image,
            },
        ),
    )
    return parse_json_content(response)


def text_extraction_agent_OS(
    image_path: str,
    *,
    graphical_input: Optional[dict] = None,
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

    # Encode image as Base64
    with open(image_path, "rb") as f:
        b64_image = base64.b64encode(f.read()).decode("utf-8")

    # Define tools for the agent
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_reactions_from_text_in_image",
                "description": "OCR image and extract chemical reactions",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "NER_from_text_in_image",
                "description": "OCR image and perform chemical named entity recognition",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image_path": {"type": "string"}
                    },
                    "required": ["image_path"],
                    "additionalProperties": False,
                }
            }
        }
    ]

    # Prompt instructing to call both tools
    prompt = """
In this input reaction graphic, there is a chemical reaction scheme template with multiple product/fragment molecular sub-images or tables, conditions, and a text description underneath (or you will receive direct text input).
Your task is to:

1. **Entity Extraction**: First call the `MolNER`("NER_from_text_in_image") tool to extract chemical identities (names, formulas, abbreviations) from the text description.
2. **Reaction Extraction**: Call the `RxnConInterpreter`("extract_reactions_from_text_in_image") tool to identify reaction types and relationships in the text description.
3. **Correction & Alignment**: 
   - Check the tool results and correct any errors.
   - Align the identified text entities with the graphical output (e.g., mapping "precatalyst G32" in text to the molecule G32 in the image).
4. **Output**: Your final output must be in JSON format. Use specific tags like `[reactant]`, `[reagent]`, `[ABBREVIATION]`, `[SMILES:...]` to annotate the original text.

### Example 1
**User Input:**
Text Input: "A visible-light-driven decarboxylative Giese addition of redox-active N-hydroxyphthalimide esters and acrylonitriles was reported by MacMillan and colleagues. Alpha-amino carboxylic acids were utilized as precursors of N-hydroxyphthalimide esters. A highly active PC photocatalyst G32 containing a bulky ligand."

**Assistant Response:**
Here is my step-by-step analysis:

**Step 1: Entity Extraction (MolNER)**
* **Observation:**
    * "N-hydroxyphthalimide esters" -> Chemical (Plural/Class)
    * "acrylonitriles" -> Chemical (Systematic Name)
    * "Alpha-amino carboxylic acids" -> Chemical (Reactant precursor)
    * "PC" -> Abbreviation
    * "G32" -> Chemical (Identifier/Reagent)
* **Action:** Tag these entities with their specific roles and types.

**Step 2: Reaction & Structure Interpretation (RxnConInterpreter)**
* **Observation:**
    * "G32" is identified as a specific precatalyst.
    * **Action:** Retrieve/Associate the SMILES for G32 from the tool/image context: `CCOC(C1=C(OC(C[C@@H]1C2=CC=CC=C2)=O)C)=O,CC1=CC(C)=CC(C)=C1[N+]2=CN3C(CO[C@@H]4[C@H]3C5=C(C=CC=C5)C4)=N2.F[B-](F)(F)F`.
    * "N-hydroxyphthalimide esters" and "acrylonitriles" are participating reactants in a decarboxylative Giese addition.

**Step 3: Text Alignment & Annotation**
* **Thought:** I need to inject the extracted tags and SMILES directly into the original text string to create the final description.
* **Action:** Construct the annotated string.

**Final JSON Output:**
```json
{
  "annotated_text": "A visible-light-driven decarboxylative Giese addition of redox-active N-hydroxyphthalimide esters [reactant][MULTIPLE] and acrylonitriles [reactant][SYSTEMATIC] was reported by MacMillan and colleagues. Alpha-amino carboxylic acids [reactant] were utilized as precursors of N-hydroxyphthalimide esters. A highly active PC [ABBREVIATION] photocatalyst Ir-44 [reagent][IDENTIFIERS][SMILES:CCOC(C1=C(OC(C[C@@H]1C2=CC=CC=C2)=O)C)=O,CC1=CC(C)=CC(C)=C1[N+]2=CN3C(CO[C@@H]4[C@H]3C5=C(C=CC=C5)C4)=N2.F[B-](F)(F)F] containing a bulky ligand."
}
```

"""

    if graphical_input:
        prompt += f"\n\nGraphical extraction results from the reaction image (use this for alignment with text entities):\n{json.dumps(graphical_input, ensure_ascii=False, indent=2)}"

    messages = [
        {"role": "system", "content": "You are the Text Extraction Agent. Your task is to extract text descriptions from chemical reaction images (or process direct text input), identify chemical entities and reactions within that text, and output a structured annotation."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
            ]
        }
    ]

    response = backend.run_tool_loop(
        LLMRequest(
            model=backend_model(backend, model_name),
            messages=messages,
            json_mode=True,
            temperature=0,
            tool_choice="auto",
        ),
        tools,
        bind_image_tools(
            image_path,
            {
                "extract_reactions_from_text_in_image": extract_reactions_from_text_in_image,
                "NER_from_text_in_image": NER_from_text_in_image,
            },
        ),
    )
    return parse_json_content(response)
