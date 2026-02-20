# -*- coding: utf-8 -*-
__title__ = "BIM Chatbot v19.13 (Refactor: _is_trench+_elec_family_match merged into _name_matches)"

from pyrevit import revit, forms
from Autodesk.Revit.DB import *
from Autodesk.Revit.DB.Plumbing import Pipe

import clr, os, json, io, re, pkgutil
clr.AddReference("System")
from System.Collections.Generic import List

doc = revit.doc
uidoc = revit.uidoc

FT_TO_MM = 304.8
FT_TO_M  = 0.3048

# ============================================================
# AI LAYER (OPTIONAL â€” SAFE FALLBACK)
# ============================================================

def ai_is_available():
    """pyRevit/IronPython-friendly package presence check."""
    return pkgutil.find_loader("openai") is not None


def _load_openai_module():
    return __import__("openai")


def _extract_ai_text(resp):
    """Supports both OpenAI legacy dict responses and v1 object responses."""
    # v0.x style: dict-like response
    try:
        return (resp["choices"][0]["message"]["content"] or "").strip()
    except:
        pass

    # v1.x style: object response
    try:
        return (resp.choices[0].message.content or "").strip()
    except:
        return ""


def ai_parse_command(cmd):
    """
    Returns dict intent or None.
    Intent format (strict JSON):
    {
      "action": "select|count|sum_length|export_qty|qa_check|list_params|export_params_excel",
      "category": "pipes|channels|plumbing|accessories|fittings",
      "filters": [{"param":"width","op":">","value":0.4}, ...]
    }
    NOTE: action export_params_excel will be exported as CSV (Excel-readable) in pyRevit.
    """
    if not ai_is_available():
        return None

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    openai = _load_openai_module()

    try:
        openai.api_key = api_key

        system_prompt = """
Convert BIM command to STRICT JSON only. No text.

Allowed actions:
select, count, sum_length, export_qty, qa_check, list_params, export_params_excel

Allowed categories:
pipes, channels, plumbing, accessories, fittings, lighting_poles, elec_trenches, panels, handholes, generators, elec_equip

Allowed params:
diameter, length, width, depth, height, thickness, elevation, pole height

Allowed operators:
>, <, =

Rules:
- "export quantities" / report / csv -> export_qty
- "export parameters" / export params / parameters to excel -> export_params_excel
- list/parameters/params (without export) -> list_params
- select -> select
- count -> count
- total length/sum/calculate -> sum_length
- check/audit/validate/qa -> qa_check
- Extract multiple filters if present (e.g. elevation > 644 and < 645).
Return JSON only.
"""
        messages = [
            {"role":"system","content":system_prompt},
            {"role":"user","content":cmd}
        ]

        # Legacy OpenAI SDK path
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=messages
            )
        except:
            # OpenAI v1+ SDK path
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=messages
            )

        txt = _extract_ai_text(resp)
        intent = json.loads(txt)
        if not isinstance(intent, dict):
            return None
        return intent

    except:
        return None


# ============================================================
# ACTIONS
# ============================================================

# ACTION_PRIORITY (v19.9) — replaces flat ACTION_WORDS.
# Each entry: token -> (priority, action)
#
# Three tiers:
#   Tier 1 (weakest)  - noun/modifier words that can appear mid-sentence
#                       e.g. "quantities", "report"
#   Tier 2 (medium)   - directional/verbal intent words
#                       e.g. "total", "sum", "audit"
#   Tier 3 (strongest)- unambiguous imperative verbs
#                       e.g. "select", "count", "export", "check"
#
# When multiple action tokens appear in one command, the highest-
# priority token wins regardless of word order.
# Equal-priority tokens: first occurrence wins (stable).
ACTION_PRIORITY = {
    # --- Tier 1: noun/modifier ---
    "quantities": (1, "export_qty"),
    "quantity":   (1, "export_qty"),
    "report":     (1, "export_qty"),
    "csv":        (1, "export_qty"),
    "excel":      (1, "export_qty"),
    # --- Tier 2: directional/verbal ---
    "total":      (2, "sum_length"),
    "sum":        (2, "sum_length"),
    "calculate":  (2, "sum_length"),
    "compute":    (2, "sum_length"),
    "generate":   (2, "export_qty"),
    "list":       (2, "list_params"),
    "params":     (2, "list_params"),
    "parameter":  (2, "list_params"),
    "parameters": (2, "list_params"),
    "audit":      (2, "qa_check"),
    "validate":   (2, "qa_check"),
    "qa":         (2, "qa_check"),
    # --- Tier 3: imperative verbs ---
    "select":     (3, "select"),
    "count":      (3, "count"),
    "export":     (3, "export_qty"),
    "check":      (3, "qa_check"),
}

OP_WORDS = {
    ">": ">", "greater": ">", "above": ">",
    "<": "<", "less": "<", "below": "<",
    "=": "=", "equal": "="
}

# ============================================================
# CATEGORY PATTERNS
# ============================================================

CATEGORY_PATTERNS = [
    ("stormwater channel", "channels"),
    ("storm water channel", "channels"),
    ("channels", "channels"),
    ("channel", "channels"),

    ("pipe accessories", "accessories"),
    ("pipe accessory", "accessories"),
    ("accessories", "accessories"),
    ("accessory", "accessories"),

    ("pipe fittings", "fittings"),
    ("fittings", "fittings"),
    ("fitting", "fittings"),

    ("manholes", "manholes"),
    ("manhole", "manholes"),
    ("valve chambers", "valve_chambers"),
    ("valve chamber", "valve_chambers"),
    ("chambers", "valve_chambers"),
    ("chamber", "valve_chambers"),
    ("plumbing", "plumbing"),       # catch-all: all plumbing fixtures

    ("pipes", "pipes"),
    ("pipe", "pipes"),

    # v19.2 Electrical Trenches
    ("electrical trench", "elec_trenches"),
    ("electrical trenches", "elec_trenches"),
    ("duct bank", "elec_trenches"),
    ("trench", "elec_trenches"),

    # v19.3 Electrical Equipment (panels, handholes, generators)
    # --- Panels ---
    ("distribution pillar", "panels"),
    ("distribution pillars", "panels"),
    ("distribution board", "panels"),
    ("distribution boards", "panels"),
    ("panels", "panels"),
    ("panel", "panels"),
    # --- Handholes ---
    ("handholes", "handholes"),
    ("handhole", "handholes"),
    # --- Generators ---
    ("electrical substation", "generators"),
    ("electrical substations", "generators"),
    ("substation", "generators"),
    ("substations", "generators"),
    ("generators", "generators"),
    ("generator", "generators"),
    # --- General (catch-all, must be LAST among elec_equip) ---
    ("electrical equipment", "elec_equip"),

    # v19.0 Street Lighting
    ("lighting poles", "lighting_poles"),
    ("lighting pole", "lighting_poles"),
    ("street lighting", "lighting_poles"),
    ("street light", "lighting_poles"),
    ("lighting fixtures", "lighting_poles"),
    ("poles", "lighting_poles"),
]

# ============================================================
# PARAM MAPS
# ============================================================

PIPE_PARAM_MAP = {
    "diameter": (BuiltInParameter.RBS_PIPE_DIAMETER_PARAM, "mm"),
    "length":   (BuiltInParameter.CURVE_ELEM_LENGTH, "m"),
}

CHANNEL_NUMERIC_PARAMS = {
    "length": BuiltInParameter.STRUCTURAL_FRAME_CUT_LENGTH,
}

# v18.1 elevation support
GENERIC_NUMERIC_KEYWORDS = {"width", "depth", "height", "thickness", "elevation"}

# ============================================================
# HELPERS
# ============================================================

def normalize(s): return (s or "").strip().lower()

def to_number(s):
    try:
        t = (s or "").strip().lower()
        t = t.replace("mm", "").replace("m", "").strip()
        return float(t)
    except:
        return None

def get_type_name(e):
    try: return e.Symbol.Name if e.Symbol else ""
    except: return ""

def get_family_name(e):
    try: return e.Symbol.Family.Name if e.Symbol else ""
    except: return ""

def compare(v,op,num):
    if v is None: return False
    if op==">": return v>num
    if op=="<": return v<num
    if op=="=": return abs(v-num)<0.001
    return True

def desktop():
    return os.path.join(os.path.expanduser("~"), "Desktop")

def _safe_str(x):
    try:
        if x is None:
            return ""
        s = str(x)
        return s
    except:
        return ""


# Optional interactive mapping cache
# key: (category, scope, logical_keyword) -> exact parameter name
PARAM_NAME_CACHE = {}


def _double_param_names(src_elem):
    names = []
    if not src_elem:
        return names
    try:
        for p in src_elem.Parameters:
            try:
                if p.StorageType == StorageType.Double:
                    n = p.Definition.Name
                    if n and n not in names:
                        names.append(n)
            except:
                pass
    except:
        pass
    return sorted(names)


def _read_exact_double_in_m(src_elem, param_name):
    if not src_elem or not param_name:
        return None
    target = normalize(param_name)
    try:
        for p in src_elem.Parameters:
            try:
                n = p.Definition.Name
                if n and normalize(n) == target and p.StorageType == StorageType.Double:
                    return p.AsDouble() * FT_TO_M
            except:
                pass
    except:
        pass
    return None


def _ask_user_param_name(e, keyword, category=None, scope="type"):
    """Ask once for an exact parameter name and cache the mapping."""
    cat = category or "generic"
    k = normalize(keyword)
    key = (cat, scope, k)

    if key in PARAM_NAME_CACHE:
        return PARAM_NAME_CACHE.get(key)

    target = None
    if scope == "type":
        try:
            target = doc.GetElement(e.GetTypeId())
        except:
            target = None
    else:
        target = e

    names = _double_param_names(target)
    if not names:
        return None

    preview = "\n".join("  - " + n for n in names[:20])
    prompt = (
        "Could not find a numeric parameter containing '{}' for category '{}'.\n\n"
        "Enter EXACT parameter name ({} parameter).\n"
        "Available names (first 20):\n{}"
    ).format(keyword, cat, scope, preview)

    user_value = forms.ask_for_string(prompt=prompt, default=names[0])
    if not user_value:
        return None

    typed = normalize(user_value)
    exact = None
    for n in names:
        if normalize(n) == typed:
            exact = n
            break

    if exact is None:
        forms.alert("Parameter '{}' not found in {} parameters for '{}'.".format(user_value, scope, cat))
        return None

    PARAM_NAME_CACHE[key] = exact
    return exact

# ============================================================
# DID YOU MEAN? (v18.9)
# Fuzzy suggestion engine - no external libraries required.
#
# Uses Jaccard similarity on character bigrams to score each
# example command against the user input, then returns the
# top 3 closest matches as a formatted hint string.
#
# Bigram example:  "select" -> {"se","el","le","ec","ct"}
# Jaccard(A,B)   = |A & B| / |A | B|   (0.0 = no match, 1.0 = identical)
# ============================================================

# Master list of all valid example commands.
# One entry per unique action+category+filter combination.
# Keep this list free of duplicates - it feeds the Did You Mean? engine.
EXAMPLE_COMMANDS = [
    # --- pipes ---
    "select pipes",
    "select pipes diameter > 300",
    "select pipes diameter > 200 and < 400",
    "select pipes elevation > 644 and < 645",
    "count pipes",
    "total length pipes",
    "total length pipes diameter > 300",
    "export quantities pipes",
    "export parameters pipes",
    "list parameters pipes",
    # --- channels ---
    "select channels",
    "select channels width > 0.5",
    "select channels width > 0.3 and < 0.6",
    "count channels",
    "total length channels",
    "export quantities channels",
    "export parameters channels",
    "list parameters channels",
    # --- plumbing: manholes ---
    "select manholes",
    "count manholes",
    "export quantities manholes",
    "export parameters manholes",
    "list parameters manholes",
    # --- plumbing: valve chambers ---
    "select valve chambers",
    "count valve chambers",
    "export quantities valve chambers",
    "export parameters valve chambers",
    # --- plumbing: all ---
    "select plumbing",
    "count plumbing",
    "export quantities plumbing",
    # --- pipe accessories & fittings ---
    "select accessories",
    "count accessories",
    "export quantities accessories",
    "export parameters accessories",
    "select fittings",
    "count fittings",
    "export quantities fittings",
    "export parameters fittings",
    # --- electrical trenches ---
    "select electrical trenches",
    "select electrical trenches width > 0.6",
    "select electrical trenches height > 0.4",
    "select electrical trenches width > 0.5 and < 1.0",
    "select electrical trenches length > 10",
    "select duct bank",
    "count electrical trenches",
    "count duct bank",
    "total length electrical trenches",
    "total length electrical trenches width > 0.6",
    "export quantities electrical trenches",
    "export quantities duct bank",
    "export parameters electrical trenches",
    # --- lighting poles ---
    "select lighting poles",
    "select lighting poles single",
    "select lighting poles double",
    "select lighting poles pole height > 8",
    "select lighting poles elevation > 644",
    "count lighting poles",
    "count lighting poles single",
    "count lighting poles double",
    "export quantities lighting poles",
    "export parameters lighting poles",
    # --- electrical equipment: panels ---
    "select panels",
    "select distribution pillars",
    "count panels",
    "export quantities panels",
    "export parameters panels",
    # --- electrical equipment: handholes ---
    "select handholes",
    "select handholes hh_width > 0.6",
    "select handholes hh_length > 0.4",
    "select handholes elevation > 644",
    "count handholes",
    "export quantities handholes",
    "export parameters handholes",
    # --- electrical equipment: generators ---
    "select generators",
    "select substation",
    "count generators",
    "export quantities generators",
    "export parameters generators",
    # --- electrical equipment: all ---
    "select electrical equipment",
    "count electrical equipment",
    "export quantities electrical equipment",
    "export parameters electrical equipment",
    # --- QA/QC ---
    "check pipes diameter",
    "check channels width",
]


def _bigrams(s):
    """Return set of character bigrams for string s."""
    s = s.lower().strip()
    if len(s) < 2:
        return set([s]) if s else set()
    return set(s[i:i+2] for i in range(len(s) - 1))


def _jaccard(a, b):
    """Jaccard similarity between two bigram sets (0.0 - 1.0)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    union        = len(a | b)
    return float(intersection) / float(union)


def did_you_mean(cmd, top_n=3):
    """
    Given a user command, return a formatted hint string showing
    the top_n most similar example commands.

    Returns empty string if the best match scores below threshold
    (avoids showing irrelevant suggestions).
    """
    THRESHOLD = 0.15   # minimum score to be worth showing
    cmd_bg = _bigrams(normalize(cmd))

    scored = []
    for ex in EXAMPLE_COMMANDS:
        score = _jaccard(cmd_bg, _bigrams(ex))
        scored.append((score, ex))

    scored.sort(key=lambda x: x[0], reverse=True)
    best = scored[:top_n]

    if best[0][0] < THRESHOLD:
        return ""   # nothing close enough - show nothing

    lines = ["Did you mean?"]
    for score, ex in best:
        if score >= THRESHOLD:
            lines.append("  > {}".format(ex))

    return "\n".join(lines)


# ============================================================
# GENERIC PARAM SEARCH (INSTANCE PARAMS)
# ============================================================

def generic_numeric_by_keyword(e,key):
    k=key.lower()
    for p in e.Parameters:
        try:
            if k in p.Definition.Name.lower():
                if p.StorageType==StorageType.Double:
                    return p.AsDouble()*FT_TO_M
        except:
            pass
    return None


def get_type_param(e, keyword, with_fallback=False, category=None, interactive=False):
    """
    Shared utility: search TYPE parameters for a Double param
    whose name contains keyword (case-insensitive).
    Returns value in metres, or None.

    Optional interactive=True:
        If no keyword match is found, asks user once for the
        exact parameter name and caches it for this session.
    """
    try:
        t = doc.GetElement(e.GetTypeId())
        if t:
            kk = keyword.lower()
            for p in t.Parameters:
                try:
                    if kk in p.Definition.Name.lower() and p.StorageType == StorageType.Double:
                        return p.AsDouble() * FT_TO_M
                except:
                    pass

            if interactive:
                exact_name = _ask_user_param_name(e, keyword, category=category, scope="type")
                if exact_name:
                    v = _read_exact_double_in_m(t, exact_name)
                    if v is not None:
                        return v
    except:
        pass

    if with_fallback:
        v = generic_numeric_by_keyword(e, keyword)
        if v is not None:
            return v

        if interactive:
            exact_name = _ask_user_param_name(e, keyword, category=category, scope="instance")
            if exact_name:
                return _read_exact_double_in_m(e, exact_name)

    return None

# ============================================================
# CLASSIC PARSER (v18.8)
# Upgraded filter extraction to support multiple conditions
# on the same parameter using regex (Option A).
#
# Examples now handled by classic parser without AI:
#   "select pipes elevation > 644 and < 645"
#   "select channels width > 0.3 and < 0.6"
#   "select pipes diameter > 200"   (single condition still works)
# ============================================================

# Regex: matches operator + number pairs anywhere in a string.
# Handles symbol ops (>, <, =) and word ops (greater, above, below, less, equal).
# The number part accepts integers and decimals e.g. 644, 0.5, 1200.75
_FILTER_PATTERN = re.compile(
    r"(>|<|=|greater|above|below|less|equal)\s*([\d]+(?:\.[\d]+)?)"
)

def parse_command(cmd):
    cmd_l = normalize(cmd)
    tokens = cmd_l.split()

    # --- Action detection ---
    # v18.3 guard: "export parameters/params" routes to param export,
    # not to export_qty, preserving backward compatibility.
    export_like = ("export" in cmd_l) or ("excel" in cmd_l) or ("csv" in cmd_l) or ("report" in cmd_l) or ("generate" in cmd_l)
    param_like  = ("parameters" in cmd_l) or ("parameter" in cmd_l) or ("params" in cmd_l) or ("param " in cmd_l) or cmd_l.endswith(" param")
    if export_like and param_like:
        action = "export_params_excel"
    else:
        # Priority-based resolution (v19.9):
        # Scan ALL tokens, keep the highest-priority action found.
        # Equal-priority ties go to the first occurrence.
        action        = None
        best_priority = -1
        for t in tokens:
            if t in ACTION_PRIORITY:
                pri, act = ACTION_PRIORITY[t]
                if pri > best_priority:
                    best_priority = pri
                    action        = act

    # --- Category detection ---
    category = None
    for phrase, cat in CATEGORY_PATTERNS:
        if phrase in cmd_l:
            category = cat
            break

    # --- Multi-condition filter extraction (v18.8 + v19.1) ---
    # Step A: check for known two-word param phrases FIRST.
    #   "pole height" must be caught before the single-token loop
    #   because neither "pole" nor "height" alone maps to this param.
    # Step B: fall back to single-token scan (original behaviour).
    filters = []
    param   = None

    # Two-word param phrases -> canonical param name
    TWO_WORD_PARAMS = [
        ("pole height", "pole height"),
        ("pole he",     "pole height"), # shorthand alias (e.g. "pole he = 0.6")
        ("pole ht",     "pole height"), # shorthand alias
        ("hh_width",    "width"),      # handhole width alias
        ("hh_length",   "length"),     # handhole length alias
        # NOTE: "duct bank" is a category phrase, NOT a param.
        # It must NOT appear here - it would intercept filter
        # extraction for commands like "select duct bank width > 0.6"
    ]

    for phrase, canonical in TWO_WORD_PARAMS:
        if phrase in cmd_l:
            param = canonical
            after_param = cmd_l[cmd_l.index(phrase) + len(phrase):]
            matches = _FILTER_PATTERN.findall(after_param)
            for raw_op, raw_val in matches:
                op  = OP_WORDS.get(raw_op)
                val = to_number(raw_val)
                if op and val is not None:
                    filters.append({"param": canonical, "op": op, "value": val})
            break

    # Step B: single-token scan (only if no two-word phrase matched)
    if param is None:
        for i, t in enumerate(tokens):
            if t in PIPE_PARAM_MAP or t in CHANNEL_NUMERIC_PARAMS or t in GENERIC_NUMERIC_KEYWORDS:
                param = t
                # Slice the command from the param keyword onward so
                # regex only matches conditions that follow the param name.
                after_param = cmd_l[cmd_l.index(t) + len(t):]
                matches = _FILTER_PATTERN.findall(after_param)

                for raw_op, raw_val in matches:
                    op  = OP_WORDS.get(raw_op)   # normalize word ops -> symbols
                    val = to_number(raw_val)
                    if op and val is not None:
                        filters.append({"param": param, "op": op, "value": val})
                break  # only one param keyword supported per command

    return action, category, filters, param, cmd_l

# ============================================================
# INTENT RESOLVER (v18.6)
# Single entry point that unifies AI path + classic parser path.
# Always returns a consistent 4-tuple:
#   (action, category, filters, cmd_l)
#
# Callers never touch ai_parse_command() or parse_command() directly.
# To add a new source (e.g. voice, REST) just add a branch here.
# ============================================================

def resolve_intent(cmd):
    """
    Resolve a raw command string into a structured intent tuple.

    Resolution order:
      1. AI parser  (if openai available + OPENAI_API_KEY set)
         -> populates action, category, filters
      2. Classic keyword parser (always available, zero dependencies)
         -> populates action, category, filters

    Returns:
        tuple: (action, category, filters, cmd_l)
               filters is always a list (empty = no filter conditions).
    """
    # --- try AI first ---
    ai_intent = None
    if ai_is_available():
        ai_intent = ai_parse_command(cmd)

    if ai_intent and isinstance(ai_intent, dict):
        action   = ai_intent.get("action")
        category = ai_intent.get("category")
        filters  = ai_intent.get("filters") or []
        cmd_l    = normalize(cmd)
        return action, category, filters, cmd_l

    # --- fall back to classic parser ---
    action, category, filters, param, cmd_l = parse_command(cmd)
    return action, category, filters, cmd_l

# ============================================================
# NUMERIC READERS
# ============================================================

def pipe_numeric(e,k):
    if k not in PIPE_PARAM_MAP:
        return None
    bip,unit=PIPE_PARAM_MAP[k]
    p=e.get_Parameter(bip)
    if not p: return None
    v=p.AsDouble()
    return v*FT_TO_MM if unit=="mm" else v*FT_TO_M

def channel_numeric(e,k):
    kk = (k or "").lower()
    if kk in CHANNEL_NUMERIC_PARAMS:
        p=e.get_Parameter(CHANNEL_NUMERIC_PARAMS[kk])
        if p: return p.AsDouble()*FT_TO_M

    # Width/Depth/Height may be modelled on Type in some families.
    if kk in ("width", "depth", "height"):
        return get_type_param(e, kk, with_fallback=True, category="channels", interactive=True)

    if kk in GENERIC_NUMERIC_KEYWORDS:
        return generic_numeric_by_keyword(e,kk)
    return None

# v18.1: plumbing numeric reader supports elevation
def plumbing_numeric(e,k):
    kk = (k or "").lower()

    if kk == "elevation":
        try:
            p = e.get_Parameter(BuiltInParameter.INSTANCE_ELEVATION_PARAM)
            if p and p.StorageType == StorageType.Double:
                return p.AsDouble() * FT_TO_M
        except:
            pass
        return None

    if kk in GENERIC_NUMERIC_KEYWORDS:
        return generic_numeric_by_keyword(e, kk)

    return None

# ============================================================
# FILTER ENGINE (v16 + v17 multi-filter)
# ============================================================

def apply_filters(elems, filters, reader):
    """
    Unified filter engine (v19.11) — replaces filter_elements + filter_multi.

    filters : list of {"param": str, "op": str, "value": float}
              An empty list means no filtering — all elements are returned.
    reader  : callable(element, param_name) -> float | None

    ALL conditions must pass for an element to be included (AND logic).
    Any condition whose reader returns None is treated as a non-match.
    """
    out = List[ElementId]()
    for e in elems:
        ok = True
        for f in filters:
            try:
                k   = f.get("param")
                op  = f.get("op")
                val = f.get("value")
                if k is None or op is None or val is None:
                    continue  # incomplete condition - skip, do not exclude
                v = reader(e, k)
                if not compare(v, op, float(val)):
                    ok = False
                    break
            except:
                ok = False
                break
        if ok:
            out.Add(e.Id)
    return out

# ============================================================
# COLLECTORS (v16 preserved)
# ============================================================

def get_pipes():
    return FilteredElementCollector(doc).OfClass(Pipe).ToElements()

def get_channels():
    elems = FilteredElementCollector(doc)\
        .OfCategory(BuiltInCategory.OST_StructuralFraming)\
        .WhereElementIsNotElementType().ToElements()
    return [e for e in elems if _name_matches(e, "channel")]

def get_plumbing():
    return FilteredElementCollector(doc)\
        .OfCategory(BuiltInCategory.OST_PlumbingFixtures)\
        .WhereElementIsNotElementType().ToElements()

def get_pipe_accessories():
    return FilteredElementCollector(doc)\
        .OfCategory(BuiltInCategory.OST_PipeAccessory)\
        .WhereElementIsNotElementType().ToElements()

def get_pipe_fittings():
    return FilteredElementCollector(doc)\
        .OfCategory(BuiltInCategory.OST_PipeFitting)\
        .WhereElementIsNotElementType().ToElements()

# ============================================================
# ELECTRICAL TRENCHES COLLECTOR + NUMERIC READER (v19.2)
#
# Matches Structural Framing families whose name contains:
#   "electrical trench", "electrical trenches",
#   "trench", "duct bank"
#
# Numeric params:
#   "length"  -> STRUCTURAL_FRAME_CUT_LENGTH (built-in instance)
#               fallback: instance param scan by keyword
#   "width"   -> Type param scan by keyword, fallback instance
#   "height"  -> Type param scan by keyword, fallback instance
# ============================================================

# Name fragments that identify an electrical trench family
_TRENCH_NAME_FRAGMENTS = [
    "electrical trench",
    "electrical trenches",
    "duct bank",
    "trench",
]

def _name_matches(e, fragments):
    """
    Shared name-fragment helper (v19.13).

    Returns True if the element's combined family+type name
    (case-insensitive) contains ANY of the given fragments.

    fragments : str or list[str]
        Pass a single string or a list of strings.

    Replaces: _is_trench, _elec_family_match, and all inline
    "keyword in (family+type).lower()" checks in collectors.
    """
    if isinstance(fragments, str):
        fragments = [fragments]
    combo = (get_family_name(e) + " " + get_type_name(e)).lower()
    return any(frag in combo for frag in fragments)


def get_electrical_trenches():
    """Returns all Structural Framing instances matching trench name fragments."""
    elems = FilteredElementCollector(doc)\
        .OfCategory(BuiltInCategory.OST_StructuralFraming)\
        .WhereElementIsNotElementType().ToElements()
    return [e for e in elems if _name_matches(e, _TRENCH_NAME_FRAGMENTS)]


def trench_numeric(e, k):
    """
    Numeric reader for electrical trenches.

    length -> STRUCTURAL_FRAME_CUT_LENGTH (built-in)
              fallback: instance param scan
    width  -> Type param scan, fallback instance
    height -> Type param scan, fallback instance
    """
    kk = (k or "").lower()

    if kk == "length":
        # Try built-in structural frame cut length first
        try:
            p = e.get_Parameter(
                BuiltInParameter.STRUCTURAL_FRAME_CUT_LENGTH)
            if p and p.StorageType == StorageType.Double:
                v = p.AsDouble()
                if v > 0:
                    return v * FT_TO_M
        except:
            pass
        # Fallback: instance param scan
        return generic_numeric_by_keyword(e, "length")

    if kk in ("width", "height"):
        return get_type_param(e, kk, with_fallback=True, category="elec_trenches", interactive=True)

    # Generic fallback for any other keyword
    return generic_numeric_by_keyword(e, kk)


def sum_lengths(elems, reader):
    """
    Unified length summer (v19.12) -- replaces:
        sum_pipe_lengths, sum_channel_lengths, sum_trench_lengths.

    Calls reader(element, "length") for each element and sums
    non-None results. Works for any category whose numeric reader
    supports the "length" keyword.
    """
    total = 0.0
    for e in elems:
        v = reader(e, "length")
        if v is not None:
            total += v
    return total


# ============================================================
# ELECTRICAL EQUIPMENT COLLECTOR + NUMERIC READER (v19.3)
#
# Three sub-categories all live under OST_ElectricalEquipment:
#   panels    -> family name contains distribution pillar/board/panel
#   handholes -> family name contains "handhole"
#   generators-> family name contains generator/substation
#   elec_equip-> all electrical equipment (no sub-filter)
#
# Handhole numeric params:
#   "hh_width"  / "width"  -> Type param containing "width"
#   "hh_length" / "length" -> Type param containing "length"
#   "elevation"            -> INSTANCE_ELEVATION_PARAM
# ============================================================

# Sub-type family name fragments
_PANEL_FRAGS    = ["distribution pillar", "distribution board", "panel"]
_HANDHOLE_FRAGS = ["handhole"]
_GEN_FRAGS      = ["generator", "electrical substation", "substation"]


def _all_elec_equip():
    """All instances in OST_ElectricalEquipment category."""
    return list(
        FilteredElementCollector(doc)
        .OfCategory(BuiltInCategory.OST_ElectricalEquipment)
        .WhereElementIsNotElementType()
        .ToElements()
    )


def get_panels():
    return [e for e in _all_elec_equip() if _name_matches(e, _PANEL_FRAGS)]


def get_handholes():
    return [e for e in _all_elec_equip() if _name_matches(e, _HANDHOLE_FRAGS)]


def get_generators():
    return [e for e in _all_elec_equip() if _name_matches(e, _GEN_FRAGS)]


def get_elec_equip_all():
    return _all_elec_equip()


def handhole_numeric(e, k):
    """
    Numeric reader for handholes.
      hh_width / width   -> Type param containing "width"
      hh_length / length -> Type param containing "length"
      elevation          -> INSTANCE_ELEVATION_PARAM
    """
    kk = (k or "").lower()

    if "width" in kk:
        return get_type_param(e, "width", category="handholes", interactive=True)

    if "length" in kk:
        return get_type_param(e, "length", category="handholes", interactive=True)

    if kk == "elevation":
        try:
            p = e.get_Parameter(
                BuiltInParameter.INSTANCE_ELEVATION_PARAM)
            if p and p.StorageType == StorageType.Double:
                return p.AsDouble() * FT_TO_M
        except:
            pass
        return generic_numeric_by_keyword(e, "elevation")

    return generic_numeric_by_keyword(e, kk)


# ============================================================
# LIGHTING POLES COLLECTOR + NUMERIC READER (v19.0)
# Family name must contain "pole" (case-insensitive).
# Type keywords for filtering: "single", "double".
# Numeric params supported:
#   "pole height"  -> Type parameter (searched by name keyword)
#   "elevation"    -> INSTANCE_FREE_HOST_OFFSET_PARAM
# ============================================================

POLE_TYPE_KEYWORDS = ["single", "double"]


def get_lighting_poles():
    """Returns all Lighting Fixture instances whose family+type name contains 'pole'."""
    elems = FilteredElementCollector(doc)\
        .OfCategory(BuiltInCategory.OST_LightingFixtures)\
        .WhereElementIsNotElementType().ToElements()
    return [e for e in elems if _name_matches(e, "pole")]


def lighting_pole_numeric(e, k):
    """
    Numeric reader for lighting poles. (v19.1)

    Supported filter keywords:
      "pole height"  -> searches Type params for any containing
                        "height" (catches Pole Height, Height,
                        Luminaire Height, etc.)
      "elevation"    -> instance elevation from host
                        (INSTANCE_FREE_HOST_OFFSET_PARAM)
    """
    kk = (k or "").lower()

    # "pole height" canonical param from parse_command two-word
    # detection, OR any keyword containing "height".
    # Search TYPE params for any parameter whose name contains
    # "height" - broad enough to catch any naming convention.
    if kk == "pole height" or "height" in kk:
        # Prefer strict "pole height" matching first so we don't silently
        # match unrelated generic height params and skip user prompt.
        val = get_type_param(
            e,
            "pole height",
            with_fallback=True,
            category="lighting_poles",
            interactive=True)
        if val is not None:
            return val

        # Last fallback: generic keyword scan for legacy families.
        return generic_numeric_by_keyword(e, "height")

    if kk == "elevation":
        try:
            p = e.get_Parameter(
                BuiltInParameter.INSTANCE_FREE_HOST_OFFSET_PARAM)
            if p and p.StorageType == StorageType.Double:
                return p.AsDouble() * FT_TO_M
        except:
            pass
        # Fallback: search instance params by keyword
        return generic_numeric_by_keyword(e, "elevation")

    return generic_numeric_by_keyword(e, kk)


# ============================================================
# CATEGORY â†’ COLLECTOR MAP (v18.2)
# ============================================================

def get_manholes():
    """Plumbing fixtures whose family+type name contains 'manhole'."""
    return [e for e in get_plumbing() if _name_matches(e, "manhole")]


def get_valve_chambers():
    """Plumbing fixtures whose family+type name contains 'valve' or 'chamber'."""
    return [e for e in get_plumbing() if _name_matches(e, ["valve", "chamber"])]


CATEGORY_COLLECTORS = {
    "pipes":          get_pipes,
    "channels":       get_channels,
    "plumbing":       get_plumbing,
    "manholes":       get_manholes,
    "valve_chambers": get_valve_chambers,
    "accessories":    get_pipe_accessories,
    "fittings":       get_pipe_fittings,
    "lighting_poles": get_lighting_poles,       # v19.0
    "elec_trenches":  get_electrical_trenches,  # v19.2
    "panels":         get_panels,               # v19.3
    "handholes":      get_handholes,            # v19.3
    "generators":     get_generators,           # v19.3
    "elec_equip":     get_elec_equip_all,       # v19.3
}

# ============================================================
# PARAMETER GROUP LABEL (v18.2)
# ============================================================

def param_group_label(pg):
    try:
        return LabelUtils.GetLabelFor(pg)
    except:
        try:
            return str(pg)
        except:
            return "Unknown Group"

# ============================================================
# LIST PARAMETERS BY GROUP (v18.2)
# ============================================================

def list_parameters_for_category(cat):
    """
    Lists INSTANCE + TYPE parameters for a sample element in the category.
    Grouped by Revit parameter group.
    """
    if cat not in CATEGORY_COLLECTORS:
        forms.alert("Category not supported for parameter listing.")
        return

    elems = CATEGORY_COLLECTORS[cat]()
    if not elems:
        forms.alert("No elements found for category: {}".format(cat))
        return

    e = elems[0]  # sample element
    groups = {}

    def add_param(p):
        try:
            name = p.Definition.Name
            grp  = param_group_label(p.Definition.ParameterGroup)
            groups.setdefault(grp, set()).add(name)
        except:
            pass

    try:
        for p in e.Parameters:
            add_param(p)
    except:
        pass

    try:
        t = doc.GetElement(e.GetTypeId())
        if t:
            for p in t.Parameters:
                add_param(p)
    except:
        pass

    lines = []
    lines.append("Category: {}".format(cat))
    lines.append("Sample Element Id: {}".format(e.Id.IntegerValue))
    lines.append("")

    for g in sorted(groups):
        lines.append("=== {} ===".format(g))
        for n in sorted(groups[g]):
            lines.append("  - " + n)
        lines.append("")

    text = "\n".join(lines)
    MAX_CHARS = 15000
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n...\n(Truncated)"

    forms.alert(text)

# ============================================================
# PARAM EXPORT (CSV that opens in Excel) â€” v18.3 ADDITIVE
# Layout matches your template concept:
# Row 1: Group headers (blanks across group span)
# Row 2: Parameter names
# Row 3+: Family, ElementId, Values
# ============================================================

def _param_to_text(p):
    try:
        st = p.StorageType

        if st == StorageType.String:
            return p.AsString() or ""

        if st == StorageType.Integer:
            # Yes/No
            try:
                if p.Definition.ParameterType == ParameterType.YesNo:
                    return "Yes" if p.AsInteger() == 1 else "No"
            except:
                pass
            return str(p.AsInteger())

        if st == StorageType.Double:
            try:
                vs = p.AsValueString()
                if vs is not None and vs != "":
                    return vs
            except:
                pass
            try:
                return str(p.AsDouble())
            except:
                return ""

        if st == StorageType.ElementId:
            eid = p.AsElementId()
            if eid and eid.IntegerValue > 0:
                try:
                    el = doc.GetElement(eid)
                    if el:
                        nm = getattr(el, "Name", None)
                        if nm:
                            return str(nm)
                except:
                    pass
                return str(eid.IntegerValue)
            return ""

    except:
        pass

    return ""


def _csv_escape(v):
    """
    Robust CSV escaping (RFC 4180):
    - strips null bytes (corrupt in any CSV reader)
    - strips tab characters (ambiguous in TSV/CSV mixed readers)
    - wraps in double-quotes if value contains comma, quote, or newline
    - doubles any internal double-quote characters
    """
    s = _safe_str(v)
    s = s.replace("\x00", "")   # remove null bytes
    s = s.replace("\t", " ")    # replace tabs with space (safe, readable)
    if any(ch in s for ch in [',', '"', '\n', '\r']):
        s = s.replace('"', '""')
        return '"' + s + '"'
    return s


def export_parameters_excel_with_values(category):
    """
    Name kept for compatibility with AI intent / command wording.
    Actual output: CSV (Excel-readable) because openpyxl is not available in pyRevit.
    """
    if category not in CATEGORY_COLLECTORS:
        forms.alert("Category not supported for parameter export.")
        return

    elems = CATEGORY_COLLECTORS[category]()
    if not elems:
        forms.alert("No elements found for category: {}".format(category))
        return

    # 1) Build union of parameter names grouped by group label
    group_to_params = {}

    def register_param(p):
        try:
            g = param_group_label(p.Definition.ParameterGroup)
            n = p.Definition.Name
            group_to_params.setdefault(g, set()).add(n)
        except:
            pass

    for e in elems:
        # instance
        try:
            for p in e.Parameters:
                register_param(p)
        except:
            pass
        # type
        try:
            t = doc.GetElement(e.GetTypeId())
            if t:
                for p in t.Parameters:
                    register_param(p)
        except:
            pass

    ordered_groups = sorted(group_to_params.keys())
    group_param_list = [(g, sorted(group_to_params[g])) for g in ordered_groups]

    # 2) Header rows
    row_group = ["Family", "ElementId"]
    row_param = ["Family", "ElementId"]

    for g, params in group_param_list:
        if not params:
            continue
        # put group label in first column of its span; blanks for the rest (Excel-like merged)
        row_group.append(g)
        for _ in range(len(params)-1):
            row_group.append("")
        # param names row
        for pname in params:
            row_param.append(pname)

    # 3) Write data rows
    rows = []
    rows.append(row_group)
    rows.append(row_param)

    for e in elems:
        fam = get_family_name(e)
        eid = e.Id.IntegerValue

        # lookup instance + type (type fills missing)
        lookup = {}

        try:
            for p in e.Parameters:
                try:
                    lookup[p.Definition.Name] = _param_to_text(p)
                except:
                    pass
        except:
            pass

        try:
            t = doc.GetElement(e.GetTypeId())
            if t:
                for p in t.Parameters:
                    try:
                        nm = p.Definition.Name
                        if nm not in lookup or lookup.get(nm, "") in ("", None):
                            lookup[nm] = _param_to_text(p)
                    except:
                        pass
        except:
            pass

        line = [_safe_str(fam), str(eid)]
        for g, params in group_param_list:
            for pname in params:
                line.append(_safe_str(lookup.get(pname, "")))
        rows.append(line)

    # 4) Save CSV  (uses unified write_csv: utf-8-sig + full escaping)
    filename = "{}_Parameters_Export.csv".format(category)
    path = os.path.join(desktop(), filename)

    # rows[0] is the group-header row, rows[1] is the param-name row,
    # rows[2:] are data rows.  write_csv expects (path, headers, data_rows)
    # so we pass the param-name row as headers and the rest as data,
    # then prepend the group-header row manually first.
    try:
        with io.open(path, "w", encoding="utf-8-sig") as f:
            # Row 1: group header (blank-padded for Excel merged-cell look)
            f.write(",".join([_csv_escape(v) for v in rows[0]]) + "\n")
        # Row 2 (param names) as header + rows 3+ as data via unified writer
        # We append so open in "a" mode for the remaining rows
        with io.open(path, "a", encoding="utf-8-sig") as f:
            for r in rows[1:]:
                f.write(",".join([_csv_escape(str(v)) for v in r]) + "\n")
    except Exception as ex:
        forms.alert("Failed to save CSV:\n{}\n\n{}".format(path, ex))
        return

    forms.alert("Parameters exported (CSV opens in Excel):\n{}".format(path))

# ============================================================
# CSV EXPORT CORE (v18.4 unified)
# - utf-8-sig: BOM tells Excel to read as UTF-8 (fixes Arabic/special chars)
# - _csv_escape applied to ALL cells (fixes commas in family/type names)
# - replaces the old bare write_csv and the inline writer in
#   export_parameters_excel_with_values
# ============================================================

def write_csv(path, headers, rows):
    """
    Unified CSV writer used by ALL exporters.
    Encoding : utf-8-sig  (Excel auto-detects UTF-8 with BOM)
    Escaping : _csv_escape on every cell (handles commas, quotes, newlines,
               null bytes, and tabs in family/type/parameter names)
    """
    try:
        with io.open(path, "w", encoding="utf-8-sig") as f:
            f.write(",".join([_csv_escape(h) for h in headers]) + "\n")
            for r in rows:
                f.write(",".join([_csv_escape(str(x)) for x in r]) + "\n")
    except Exception as ex:
        forms.alert("Failed to save CSV:\n{}\n\n{}".format(path, ex))
        raise

# ============================================================
# EXPORTERS (v16 preserved)
# ============================================================

def export_pipe_quantities():
    groups={}
    for p in get_pipes():
        d=p.get_Parameter(BuiltInParameter.RBS_PIPE_DIAMETER_PARAM)
        l=p.get_Parameter(BuiltInParameter.CURVE_ELEM_LENGTH)
        if not d or not l: continue
        dia=int(round(d.AsDouble()*FT_TO_MM))
        lm=l.AsDouble()*FT_TO_M
        groups[dia]=groups.get(dia,0)+lm

    rows=[[k,round(groups[k],3)] for k in sorted(groups)]
    path=os.path.join(desktop(),"Pipes_By_Diameter.csv")
    write_csv(path,["Diameter_mm","Total_Length_m"],rows)
    return path

def export_channel_quantities():
    groups={}
    for c in get_channels():
        w=generic_numeric_by_keyword(c,"width")
        d=generic_numeric_by_keyword(c,"depth")
        l=c.get_Parameter(BuiltInParameter.STRUCTURAL_FRAME_CUT_LENGTH)
        if w is None or d is None or not l: continue
        key=(round(w,3),round(d,3))
        groups[key]=groups.get(key,0)+(l.AsDouble()*FT_TO_M)

    rows=[[k[0],k[1],round(groups[k],3)] for k in sorted(groups)]
    path=os.path.join(desktop(),"Channels_By_WidthDepth.csv")
    write_csv(path,["Width_m","Depth_m","Total_Length_m"],rows)
    return path

def export_manhole_by_family():
    groups={}
    for e in get_plumbing():
        f=get_family_name(e) or "Unknown_Family"
        groups[f]=groups.get(f,0)+1

    rows=[[k,groups[k]] for k in sorted(groups)]
    path=os.path.join(desktop(),"Manholes_By_Family.csv")
    write_csv(path,["Family_Name","Count"],rows)
    return path

def export_valve_chambers_by_type():
    elems = get_valve_chambers()

    groups={}
    for e in elems:
        t=get_type_name(e) or "Unknown_Type"
        groups[t]=groups.get(t,0)+1

    rows=[[k,groups[k]] for k in sorted(groups)]
    path=os.path.join(desktop(),"ValveChambers_By_Type.csv")
    write_csv(path,["Type_Name","Count"],rows)
    return path


def export_plumbing_all():
    """
    Combined plumbing export: runs both sub-exporters and
    returns both file paths joined as a single string.
    Used when category="plumbing" (bare "export quantities plumbing"
    or in the export-all fallback).
    """
    path1 = export_manhole_by_family()
    path2 = export_valve_chambers_by_type()
    return path1 + "\n" + path2


def export_pipe_accessories_by_family():
    groups={}
    for e in get_pipe_accessories():
        f=get_family_name(e) or "Unknown_Family"
        groups[f]=groups.get(f,0)+1

    rows=[[k,groups[k]] for k in sorted(groups)]
    path=os.path.join(desktop(),"PipeAccessories_By_Family.csv")
    write_csv(path,["Family_Name","Count"],rows)
    return path

def export_pipe_fittings_by_type():
    groups={}
    for e in get_pipe_fittings():
        t=get_type_name(e) or "Unknown_Type"
        groups[t]=groups.get(t,0)+1

    rows=[[k,groups[k]] for k in sorted(groups)]
    path=os.path.join(desktop(),"PipeFittings_By_Type.csv")
    write_csv(path,["Type_Name","Count"],rows)
    return path

# ============================================================
# LIGHTING POLES EXPORTER (v19.0)
# Groups by Type name, then by Single/Double keyword in type name.
# Columns: Family | Type | Category (Single/Double) | Pole Height (m) | Count
# ============================================================

def export_lighting_poles_by_type():
    groups = {}
    for e in get_lighting_poles():
        fam  = get_family_name(e) or "Unknown_Family"
        typ  = get_type_name(e)   or "Unknown_Type"

        # Detect Single / Double from type name
        typ_l = typ.lower()
        if "double" in typ_l:
            cat_label = "Double"
        elif "single" in typ_l:
            cat_label = "Single"
        else:
            cat_label = "Other"

        # Read Pole Height from Type parameter
        ph = get_type_param(e, "height")
        ph_str = "{:.2f}".format(ph) if ph is not None else ""

        key = (fam, typ, cat_label, ph_str)
        groups[key] = groups.get(key, 0) + 1

    rows = [[k[0], k[1], k[2], k[3], groups[k]]
            for k in sorted(groups)]
    path = os.path.join(desktop(), "LightingPoles_By_Type.csv")
    write_csv(path,
              ["Family_Name", "Type_Name", "Category",
               "Pole_Height_m", "Count"],
              rows)
    return path


# ============================================================
# ELECTRICAL TRENCH EXPORTER (v19.2)
# Groups by Family + Type + Width + Height.
# Reads Width & Height from Type params; Length from built-in.
# Columns: Family | Type | Width_m | Height_m | Total_Length_m
# ============================================================

def export_trench_quantities():
    groups = {}  # key: (family, type, width_str, height_str)
                 # value: total length in metres

    for e in get_electrical_trenches():
        fam  = get_family_name(e) or "Unknown_Family"
        typ  = get_type_name(e)   or "Unknown_Type"

        w = get_type_param(e, "width",  with_fallback=True)
        h = get_type_param(e, "height", with_fallback=True)
        l = trench_numeric(e, "length")

        w_str = "{:.3f}".format(w) if w is not None else ""
        h_str = "{:.3f}".format(h) if h is not None else ""
        l_val = l if l is not None else 0.0

        key = (fam, typ, w_str, h_str)
        groups[key] = groups.get(key, 0.0) + l_val

    rows = [
        [k[0], k[1], k[2], k[3], round(groups[k], 3)]
        for k in sorted(groups)
    ]
    path = os.path.join(desktop(), "ElecTrenches_Quantities.csv")
    write_csv(path,
              ["Family_Name", "Type_Name",
               "Width_m", "Height_m", "Total_Length_m"],
              rows)
    return path


# ============================================================
# ELECTRICAL EQUIPMENT EXPORTERS (v19.3)
# ============================================================

def export_panels_quantities():
    """Panels grouped by Family + Type."""
    groups = {}
    for e in get_panels():
        key = (get_family_name(e) or "Unknown_Family",
               get_type_name(e)   or "Unknown_Type")
        groups[key] = groups.get(key, 0) + 1
    rows = [[k[0], k[1], groups[k]] for k in sorted(groups)]
    path = os.path.join(desktop(), "Panels_Quantities.csv")
    write_csv(path, ["Family_Name", "Type_Name", "Count"], rows)
    return path


def export_handholes_quantities():
    """Handholes grouped by Family + Type + HH_Width + HH_Length."""
    groups = {}
    for e in get_handholes():
        fam = get_family_name(e) or "Unknown_Family"
        typ = get_type_name(e)   or "Unknown_Type"
        w = get_type_param(e, "width")
        h = get_type_param(e, "length")
        w_str = "{:.3f}".format(w) if w is not None else ""
        l_str = "{:.3f}".format(h) if h is not None else ""
        key = (fam, typ, w_str, l_str)
        groups[key] = groups.get(key, 0) + 1
    rows = [[k[0], k[1], k[2], k[3], groups[k]]
            for k in sorted(groups)]
    path = os.path.join(desktop(), "Handholes_Quantities.csv")
    write_csv(path,
              ["Family_Name", "Type_Name",
               "HH_Width_m", "HH_Length_m", "Count"],
              rows)
    return path


def export_generators_quantities():
    """Generators grouped by Family + Type."""
    groups = {}
    for e in get_generators():
        key = (get_family_name(e) or "Unknown_Family",
               get_type_name(e)   or "Unknown_Type")
        groups[key] = groups.get(key, 0) + 1
    rows = [[k[0], k[1], groups[k]] for k in sorted(groups)]
    path = os.path.join(desktop(), "Generators_Quantities.csv")
    write_csv(path, ["Family_Name", "Type_Name", "Count"], rows)
    return path


def export_elec_equip_quantities():
    """All electrical equipment: Family + Type + Sub-Category + Count."""
    groups = {}
    for e in _all_elec_equip():
        fam = get_family_name(e) or "Unknown_Family"
        typ = get_type_name(e)   or "Unknown_Type"
        # Detect sub-category label
        if _name_matches(e, _HANDHOLE_FRAGS):
            sub = "Handhole"
        elif _name_matches(e, _PANEL_FRAGS):
            sub = "Panel"
        elif _name_matches(e, _GEN_FRAGS):
            sub = "Generator"
        else:
            sub = "Other"
        key = (sub, fam, typ)
        groups[key] = groups.get(key, 0) + 1
    rows = [[k[0], k[1], k[2], groups[k]] for k in sorted(groups)]
    path = os.path.join(desktop(), "ElecEquipment_Quantities.csv")
    write_csv(path,
              ["Sub_Category", "Family_Name", "Type_Name", "Count"],
              rows)
    return path


# ============================================================
# EXPORT DISPATCH MAP — population (v19.5)
# All exporter functions are now defined so we can reference them.
# ============================================================

EXPORT_MAP = {
    "pipes":          export_pipe_quantities,
    "channels":       export_channel_quantities,
    # plumbing: combined exporter runs manholes + valve chambers
    "plumbing":       export_plumbing_all,
    "manholes":       export_manhole_by_family,
    "valve_chambers": export_valve_chambers_by_type,
    "accessories":    export_pipe_accessories_by_family,
    "fittings":       export_pipe_fittings_by_type,
    "lighting_poles": export_lighting_poles_by_type,
    "elec_trenches":  export_trench_quantities,
    "panels":         export_panels_quantities,
    "handholes":      export_handholes_quantities,
    "generators":     export_generators_quantities,
    # elec_equip (catch-all) exports all sub-types in one CSV
    "elec_equip":     export_elec_equip_quantities,
}

# ============================================================
# QA/QC RULES (v18)
# ============================================================

def qa_pipes_missing_diameter():
    out=List[ElementId]()
    for p in get_pipes():
        d=p.get_Parameter(BuiltInParameter.RBS_PIPE_DIAMETER_PARAM)
        if not d or d.AsDouble()==0:
            out.Add(p.Id)
    return out

def qa_channels_missing_width():
    out=List[ElementId]()
    for c in get_channels():
        if generic_numeric_by_keyword(c,"width") is None:
            out.Add(c.Id)
    return out

# ============================================================
# MAIN (v18.7)
# All execution logic lives here.
# - No more raise SystemExit scattered through the code
# - return exits cleanly at any decision point
# - top-level try/except catches unexpected errors with a
#   readable alert instead of a raw IronPython traceback
# ============================================================

def main():

    # ----------------------------------------------------------
    # 1. GET INPUT
    # ----------------------------------------------------------
    cmd = forms.ask_for_string(prompt="Enter command", default="export quantities")
    if not cmd:
        return  # user cancelled dialog - exit silently

    # ----------------------------------------------------------
    # 2. RESOLVE INTENT
    # Single call handles AI + classic parser (v18.6)
    # ----------------------------------------------------------
    action, category, filters, cmd_l = resolve_intent(cmd)

    # ----------------------------------------------------------
    # 3. ROUTE BY ACTION
    # ----------------------------------------------------------

    # --- Parameter Export (CSV / Excel) ---
    if action == "export_params_excel":
        if not category:
            forms.alert("Please specify category.\n\nExamples:\n  export parameters pipes\n  export params manholes")
            return
        export_parameters_excel_with_values(category)
        return

    # --- Parameter List ---
    if action == "list_params":
        if not category:
            forms.alert("Please specify category.\n\nExamples:\n  list parameters pipes\n  params channels")
            return
        list_parameters_for_category(category)
        return

    # --- Quantity Export ---
    if action == "export_qty":
        msgs = []

        if category and category in EXPORT_MAP:
            # Category resolved cleanly: call its dedicated exporter.
            msgs.append(EXPORT_MAP[category]())
        else:
            # No specific category ("export quantities" with no qualifier)
            # -> export everything: one file per category.
            for cat, fn in EXPORT_MAP.items():
                # Skip sub-categories when exporting all:
                # - elec_equip covers panels/handholes/generators
                # - plumbing covers manholes/valve_chambers
                if cat in ("panels", "handholes", "generators",
                           "manholes", "valve_chambers"):
                    continue
                try:
                    msgs.append(fn())
                except Exception as ex:
                    msgs.append("  [skipped {}: {}]".format(cat, ex))

        forms.alert("Exported:\n" + "\n".join(msgs))
        return

    # --- QA / QC Check ---
    if action == "qa_check":
        if "pipe" in cmd_l and "diameter" in cmd_l:
            results = qa_pipes_missing_diameter()
        elif "channel" in cmd_l and "width" in cmd_l:
            results = qa_channels_missing_width()
        else:
            forms.alert("QA rule not recognized.\n\nSupported rules:\n  check pipes diameter\n  check channels width")
            return

        uidoc.Selection.SetElementIds(results)
        forms.alert("QA Issues Found: {}".format(results.Count))
        return

    # --- Query / Select / Count / Sum (need category resolved first) ---
    if category == "pipes":
        elems   = get_pipes()
        results = apply_filters(elems, filters, pipe_numeric)

    elif category == "channels":
        elems   = get_channels()
        results = apply_filters(elems, filters, channel_numeric)

    elif category == "manholes":
        elems   = get_manholes()
        results = apply_filters(elems, filters, plumbing_numeric)

    elif category == "valve_chambers":
        elems   = get_valve_chambers()
        results = apply_filters(elems, filters, plumbing_numeric)

    elif category == "plumbing":  # catch-all: all plumbing fixtures
        elems   = get_plumbing()
        results = apply_filters(elems, filters, plumbing_numeric)

    elif category == "accessories":
        results = List[ElementId]([e.Id for e in get_pipe_accessories()])

    elif category == "fittings":
        results = List[ElementId]([e.Id for e in get_pipe_fittings()])

    elif category == "elec_trenches":
        elems = get_electrical_trenches()
        if not elems:
            forms.alert(
                "No Electrical Trench families found in this model.\n\n"
                "Family name must contain: electrical trench, "
                "trench, or duct bank.")
            return
        results = apply_filters(elems, filters, trench_numeric)

    elif category == "panels":
        elems = get_panels()
        if not elems:
            forms.alert(
                "No Panel families found.\n\n"
                "Family name must contain: distribution pillar, "
                "distribution board, or panel.")
            return
        results = List[ElementId]([e.Id for e in elems])

    elif category == "handholes":
        elems = get_handholes()
        if not elems:
            forms.alert(
                "No Handhole families found.\n\n"
                "Family name must contain: handhole.")
            return
        results = apply_filters(elems, filters, handhole_numeric)

    elif category == "generators":
        elems = get_generators()
        if not elems:
            forms.alert(
                "No Generator/Substation families found.\n\n"
                "Family name must contain: generator, "
                "electrical substation, or substation.")
            return
        results = List[ElementId]([e.Id for e in elems])

    elif category == "elec_equip":
        elems = get_elec_equip_all()
        if not elems:
            forms.alert("No Electrical Equipment found in this model.")
            return
        results = List[ElementId]([e.Id for e in elems])

    elif category == "lighting_poles":
        elems = get_lighting_poles()

        # --- Guard: no poles found at all ---
        if not elems:
            forms.alert(
                "No Lighting Pole families found in this model.\n\n"
                "Make sure your Lighting Fixture family name contains 'pole'.")
            return

        # --- Optional Type filter: Single / Double ---
        # Checks the Revit Type name for the keyword.
        type_kw = None
        for kw in POLE_TYPE_KEYWORDS:
            if kw in cmd_l:
                type_kw = kw
                break

        if type_kw:
            filtered_by_type = [e for e in elems
                                if type_kw in get_type_name(e).lower()]
            # Guard: type keyword found in command but no matching types
            if not filtered_by_type:
                all_types = sorted(set(
                    get_type_name(e) for e in elems if get_type_name(e)))
                forms.alert(
                    "No lighting poles found with type containing '{}'\n\n"
                    "Available types in model:\n{}".format(
                        type_kw,
                        "\n".join("  - " + t for t in all_types[:20])))
                return
            elems = filtered_by_type

        results = apply_filters(elems, filters, lighting_pole_numeric)

    else:
        hint = did_you_mean(cmd)
        msg  = "Command not understood: '{}'".format(cmd)
        if hint:
            msg += "\n\n" + hint
        else:
            msg += ("\n\nTry:\n"
                    "  select pipes diameter > 300\n"
                    "  count manholes\n"
                    "  export quantities\n"
                    "  check pipes diameter")
        forms.alert(msg)
        return

    # ----------------------------------------------------------
    # 4. EXECUTE QUERY ACTION
    # ----------------------------------------------------------
    if action == "select":
        uidoc.Selection.SetElementIds(results)
        forms.alert("Selected: {}".format(results.Count))

    elif action == "count":
        forms.alert("Count = {}".format(results.Count))

    elif action == "sum_length":
        # Dispatch table: category -> its numeric reader.
        # To support a new category just add one entry here.
        LENGTH_READERS = {
            "pipes":         pipe_numeric,
            "channels":      channel_numeric,
            "elec_trenches": trench_numeric,
        }
        if category not in LENGTH_READERS:
            unsupported_msg = (
                "Length sum is not supported for '{}'.\n\n"
                "Supported: pipes, channels, electrical trenches."
            ).format(category)
            forms.alert(unsupported_msg)
        else:
            elems  = [doc.GetElement(i) for i in results]
            reader = LENGTH_READERS[category]
            total  = sum_lengths(elems, reader)
            _LENGTH_LABELS = {
                "pipes":         "Pipe",
                "channels":      "Channel",
                "elec_trenches": "Electrical Trench",
            }
            label = _LENGTH_LABELS.get(category, category)
            forms.alert("Total {} Length = {:.2f} m".format(label, total))

    else:
        hint = did_you_mean(cmd)
        msg  = "Action not recognized: '{}' for category '{}'".format(action or "?", category)
        if hint:
            msg += "\n\n" + hint
        else:
            msg += "\n\nTry: select, count, or total length"
        forms.alert(msg)


# ============================================================
# ENTRY POINT
# Wraps main() in a top-level try/except so any unexpected
# error shows a clean alert instead of a raw IronPython traceback.
# ============================================================

try:
    main()
except Exception as _err:
    forms.alert("Unexpected error:\n\n{}".format(_err))
