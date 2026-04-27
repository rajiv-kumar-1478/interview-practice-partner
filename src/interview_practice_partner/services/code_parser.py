"""Code parser for detecting solution formats and extracting code blocks.

This module provides utilities for:
- Detecting whether a response is code, pseudocode, or explanation
- Extracting code blocks from markdown-formatted or plain text responses
- Handling multiple code blocks and selecting the most relevant one
"""

from __future__ import annotations

import re
from typing import Optional

from interview_practice_partner.domain.enums import SolutionFormat


# ---------------------------------------------------------------------------
# Code fence detection patterns
# ---------------------------------------------------------------------------

# Matches markdown code fences with optional language specifier
# Examples: ```python, ```java, ```javascript, ```
_CODE_FENCE_PATTERN = re.compile(
    r"```(?:[a-zA-Z0-9+#-]*)\s*\n(.*?)\n```",
    re.DOTALL | re.MULTILINE,
)

# Matches markdown code fences and captures the language specifier
_CODE_FENCE_WITH_LANG_PATTERN = re.compile(
    r"```([a-zA-Z0-9+#-]*)\s*\n(.*?)\n```",
    re.DOTALL | re.MULTILINE,
)

# Common programming language keywords that indicate actual code
_CODE_KEYWORDS = frozenset([
    "def", "function", "class", "return", "if", "else", "elif", "for", "while",
    "import", "from", "const", "let", "var", "public", "private", "static",
    "void", "int", "string", "bool", "float", "double", "char", "struct",
    "interface", "extends", "implements", "new", "this", "self", "super",
    "try", "catch", "finally", "throw", "throws", "async", "await", "yield",
    "lambda", "=>", "->", "null", "nullptr", "None", "true", "false", "True", "False",
])

# Pseudocode indicators (natural language mixed with structure)
_PSEUDOCODE_INDICATORS = frozenset([
    "step", "initialize", "iterate", "check", "compare", "update",
    "repeat", "until", "each", "every", "all", "some", "find",
])

# Syntax characters that indicate code rather than natural language
_CODE_SYNTAX_CHARS = frozenset([
    "{", "}", "[", "]", "(", ")", ";", "==", "!=", "<=", ">=",
    "&&", "||", "++", "--", "+=", "-=", "*=", "/=",
])

# Language-specific keyword patterns for language detection
_LANGUAGE_KEYWORDS = {
    "python": frozenset(["def", "import", "from", "class", "self", "None", "True", "False", "elif", "lambda", "yield", "with", "as", "pass", "raise", "except", "finally"]),
    "javascript": frozenset(["function", "const", "let", "var", "=>", "async", "await", "console", "document", "window", "export", "import", "require"]),
    "typescript": frozenset(["interface", "type", "enum", "namespace", "readonly", "public", "private", "protected", "implements", "extends"]),
    "java": frozenset(["public", "private", "protected", "static", "void", "class", "interface", "extends", "implements", "new", "package", "import", "throws"]),
    "cpp": frozenset(["#include", "std::", "cout", "cin", "endl", "namespace", "using", "template", "typename", "nullptr", "virtual", "override"]),
    "c": frozenset(["#include", "printf", "scanf", "malloc", "free", "struct", "typedef", "sizeof", "NULL"]),
    "csharp": frozenset(["using", "namespace", "public", "private", "protected", "static", "void", "class", "interface", "var", "async", "await", "Task"]),
    "go": frozenset(["func", "package", "import", "type", "struct", "interface", "go", "defer", "chan", "range", "make"]),
    "rust": frozenset(["fn", "let", "mut", "impl", "trait", "struct", "enum", "pub", "use", "mod", "match", "Some", "None", "Ok", "Err"]),
    "ruby": frozenset(["def", "end", "class", "module", "require", "attr_accessor", "attr_reader", "attr_writer", "puts", "gets", "do"]),
    "swift": frozenset(["func", "var", "let", "class", "struct", "enum", "protocol", "extension", "import", "guard", "defer"]),
    "kotlin": frozenset(["fun", "val", "var", "class", "object", "interface", "data", "sealed", "companion", "when", "is"]),
}


def parse_solution_format(response_text: str) -> SolutionFormat:
    """Determine if response is code, pseudocode, or explanation.
    
    Detection logic:
    1. If markdown code fences are present → CODE
    2. If indentation patterns + code keywords → CODE
    3. If pseudocode indicators + some structure → PSEUDOCODE
    4. Otherwise → EXPLANATION
    
    Args:
        response_text: The user's response text to analyze.
    
    Returns:
        SolutionFormat enum indicating the detected format.
    """
    if not response_text or not response_text.strip():
        return SolutionFormat.EXPLANATION
    
    text = response_text.strip()
    lower_text = text.lower()
    
    # Check for markdown code fences
    if _CODE_FENCE_PATTERN.search(text):
        return SolutionFormat.CODE
    
    # Count code indicators
    code_keyword_count = sum(1 for kw in _CODE_KEYWORDS if kw in lower_text)
    pseudocode_indicator_count = sum(1 for ind in _PSEUDOCODE_INDICATORS if ind in lower_text)
    syntax_char_count = sum(1 for char in _CODE_SYNTAX_CHARS if char in text)
    
    # Check for indentation patterns (4+ spaces or tabs at line start)
    lines = text.split("\n")
    indented_lines = sum(1 for line in lines if line.startswith("    ") or line.startswith("\t"))
    indentation_ratio = indented_lines / len(lines) if lines else 0
    
    # Decision logic
    # Strong code indicators: multiple keywords + syntax + indentation
    if code_keyword_count >= 3 and (syntax_char_count >= 2 or indentation_ratio > 0.3):
        return SolutionFormat.CODE
    
    # Pseudocode indicators: natural language structure words
    # Allow pseudocode even without code keywords if there are enough pseudocode indicators
    if pseudocode_indicator_count >= 2:
        return SolutionFormat.PSEUDOCODE
    
    # Weak code indicators: some keywords but mostly natural language
    if code_keyword_count >= 2 and syntax_char_count >= 1:
        return SolutionFormat.CODE
    
    # Default to explanation if no strong indicators
    return SolutionFormat.EXPLANATION


def extract_code_blocks(text: str) -> list[str]:
    """Extract all code blocks from text.
    
    Handles:
    - Markdown code fences (```language ... ```)
    - Plain indented code blocks (4+ spaces or tabs)
    - Inline code snippets (backticks) - only as fallback when no blocks found
    
    Args:
        text: The text to extract code blocks from.
    
    Returns:
        List of extracted code blocks (strings), empty if none found.
    """
    if not text or not text.strip():
        return []
    
    blocks: list[str] = []
    
    # Extract markdown code fences
    fence_matches = _CODE_FENCE_PATTERN.findall(text)
    blocks.extend(fence_matches)
    
    # If we found fenced blocks, return those (they're most explicit)
    if blocks:
        return [block.strip() for block in blocks if block.strip()]
    
    # Otherwise, try to extract indented blocks
    lines = text.split("\n")
    current_block: list[str] = []
    
    for line in lines:
        # Check if line is indented (4+ spaces or tab)
        if line.startswith("    ") or line.startswith("\t"):
            # Remove leading indentation
            dedented = line.lstrip()
            current_block.append(dedented)
        else:
            # Non-indented line - end current block if any
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
    
    # Don't forget the last block
    if current_block:
        blocks.append("\n".join(current_block))
    
    # If we found indented blocks, return those
    if blocks:
        return [block.strip() for block in blocks if block.strip()]
    
    # As a last resort, extract inline code snippets (single backticks)
    # Pattern matches `code` but not ``` (which would be fence markers)
    inline_pattern = re.compile(r'(?<!`)`([^`\n]+)`(?!`)', re.MULTILINE)
    inline_matches = inline_pattern.findall(text)
    
    # Only return inline snippets if they look like code (have code keywords/syntax)
    # This avoids extracting random backtick-wrapped text
    code_like_snippets = []
    for snippet in inline_matches:
        snippet_lower = snippet.lower()
        # Check if snippet has code indicators
        has_code_keyword = any(kw in snippet_lower for kw in _CODE_KEYWORDS)
        has_syntax = any(char in snippet for char in _CODE_SYNTAX_CHARS)
        
        if has_code_keyword or has_syntax or len(snippet.split()) <= 3:
            # Include if it has code indicators or is short (likely a variable/function name)
            code_like_snippets.append(snippet.strip())
    
    return code_like_snippets


def get_primary_code_block(text: str) -> Optional[str]:
    """Get the primary code block from text (last complete block).
    
    When multiple code blocks are present, returns the last one as it's
    most likely the user's final solution.
    
    Args:
        text: The text to extract code from.
    
    Returns:
        The primary code block string, or None if no code blocks found.
    """
    blocks = extract_code_blocks(text)
    if not blocks:
        return None
    
    # Return the last block (most recent/final solution)
    return blocks[-1]


def detect_programming_language(text: str) -> Optional[str]:
    """Detect the programming language of code text.
    
    First checks for explicit language specifier in markdown code fences.
    If not found, uses keyword-based heuristics to detect the language.
    
    Args:
        text: The code text to analyze.
    
    Returns:
        Language name (lowercase) if detected, None otherwise.
        Possible values: "python", "javascript", "typescript", "java", 
        "cpp", "c", "csharp", "go", "rust", "ruby", "swift", "kotlin"
    """
    if not text or not text.strip():
        return None
    
    # First, check for explicit language in code fence
    fence_match = _CODE_FENCE_WITH_LANG_PATTERN.search(text)
    if fence_match:
        lang = fence_match.group(1).lower().strip()
        if lang:
            # Normalize common variations
            lang_map = {
                "py": "python",
                "js": "javascript",
                "ts": "typescript",
                "c++": "cpp",
                "cs": "csharp",
                "rb": "ruby",
                "kt": "kotlin",
            }
            return lang_map.get(lang, lang)
    
    # Keyword-based detection
    lower_text = text.lower()
    
    # Count keyword matches for each language
    language_scores: dict[str, int] = {}
    for lang, keywords in _LANGUAGE_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower_text)
        if score > 0:
            language_scores[lang] = score
    
    if not language_scores:
        return None
    
    # Return language with highest score
    # Require at least 2 keyword matches to avoid false positives
    best_lang = max(language_scores.items(), key=lambda x: x[1])
    if best_lang[1] >= 2:
        return best_lang[0]
    
    return None
