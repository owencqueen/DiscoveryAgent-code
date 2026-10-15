CLAUDEMD_prompt = """Analyze this Python codebase comprehensively and create a detailed CLAUDE.md file that will help future ClaudeCode iterations understand the project without having to explore from scratch. Focus specifically on Python modules, functions, and classes you've written. Follow these specific instructions:

## Analysis Requirements

1. **Scan the directory structure** - Map out all .py files and their organization
2. **Catalog all relevant functions and classes** - Extract function/class names, purposes, parameters, return types, and exact file locations. Catalog only functions that you believe will be relevant for future analyses.
3. **Document import paths** - Show exactly how to import each function/class from its location
4. **Map module dependencies** - Understand how Python files import from each other within the project
5. **Identify reusable utilities** - Find functions and classes that can be reused across different parts of the project
6. **Note Python-specific patterns** - Identify coding conventions, module organization, and project structure
7. **Track what YOU created** - Focus specifically on code written by ClaudeCode, not external dependencies
8. **Document relative imports** - Since no local pip installs are allowed, show proper relative import syntax

## CLAUDE.md Structure to Generate

Create a CLAUDE.md file with these sections:

### Project Overview
- Brief description of what this Python project does
- Main Python modules and their roles

### # Import Reference Guide
**Quick copy-paste imports for commonly used functions:**
```python
# Utility functions
from src.utils.helpers import helper_function, AnotherHelper

# Data models
from src.models.data_models import DataModel, ProcessingClass

# Main components
from src.main import main_function
```

### Common Utility Patterns
- Frequently used helper functions
- Common class inheritance patterns
- Standard error handling approaches
- Logging patterns used


## Reusable Code Inventory
**Functions/classes that can be reused in new features:**
- `utility_function()` in `src/utils/helpers.py` - Use for [specific purpose]
- `DataProcessor` class in `src/models/processor.py` - Use for [specific purpose]
- Common patterns that can be copied and adapted

## Execution Instructions

1. **Focus on YOUR code** - Only document functions and classes created by ClaudeCode iterations
2. **Precise import paths** - Always show exact import statements needed
3. **Function signatures** - Include parameter types and return types when available
4. **Location tracking** - Always specify which file contains each function/class
5. **Reusability focus** - Emphasize code that can be reused to avoid duplication
6. **Python-specific** - Focus on Python modules, packages, and import mechanisms
7. **No external tools** - Don't document build tools, just the Python code structure

## Output Format

Write the CLAUDE.md file directly to the project root. Focus on making it a comprehensive reference for Python code written by ClaudeCode.

After creating the file, provide a brief summary of:
- Total Python files analyzed
- Number of functions/classes cataloged  
- Key reusable components identified
- Import patterns discovered

Remember: This should be an inventory of all Python code YOU have written, with exact instructions on how to import and use each piece.

IMPORTANT: Make sure this document is concise and informative. If a CLAUDE.md file exists, add to it, don't remove anything unless the information is no longer relevant.
IMPORTANT: You must write the file to this exact location: {claude_md_path}"""

DOCUMENTATION_PROMPT = '''You are a skilled coding agent tasked with maintaining a detailed DOCUMENTATION.md file that serves as complete technical documentation for this project.
You are also responsible for being able to search the documentation and provide guidance on where the proper code is located and what has previously been done. 
Focus on all Python modules, functions, and classes that have been written in this turn (not those already in the documentation, if applicable). Follow these specific instructions:

## Analysis Requirements

1. **Scan the directory structure** - Map out all .py files and their organization
2. **Catalog ALL functions and classes** - Extract function/class names, purposes, parameters, return types, and exact file locations with comprehensive detail
3. **Document import paths** - Show exactly how to import each function/class from its location
4. **Map module dependencies** - Understand and document how Python files import from each other within the project
5. **Document API interfaces** - Detail all public-facing functions and classes with examples
6. **Track implementation details** - Document algorithms, data structures, and design patterns used
7. **Include usage examples** - Provide concrete code examples for major components
8. **Document relative imports** - Since no local pip installs are allowed, show proper relative import syntax

## DOCUMENTATION.md Structure to Generate

Create a DOCUMENTATION.md file with these sections:

### Project Overview
- Comprehensive description of what this Python project does
- Architecture overview and design philosophy
- Main Python modules and their detailed roles
- System requirements and dependencies

### Module Reference
For each major module/file:
- **Purpose** - What this module does
- **Location** - Exact file path
- **Key Classes** - Detailed class documentation including:
  - Class purpose and responsibilities
  - Constructor parameters with types
  - All public methods with signatures
  - Attributes and properties
  - Inheritance relationships
- **Key Functions** - Detailed function documentation including:
  - Function purpose
  - Parameters (name, type, description, default values)
  - Return type and description
  - Raises (exceptions that can be thrown)
  - Usage examples
- **Dependencies** - What this module imports

### API Reference
Complete alphabetical listing of all functions and classes:
```python
function_name(param1: type, param2: type = default) -> return_type
    """
    Detailed description
    
    Args:
        param1: Description
        param2: Description
        
    Returns:
        Description
        
    Raises:
        ExceptionType: When this happens
    """
```

### Design Patterns and Conventions
- Code organization patterns used
- Naming conventions
- Error handling strategies
- Logging approaches
- Common idioms and patterns

### Usage Examples
- Complete working examples for main functionality
- Common workflows with code
- Integration examples

### Development Guide
- How to extend the codebase
- Where to add new features
- Testing approaches
- Common pitfalls to avoid

## Execution Instructions

1. **Be comprehensive** - Document every function and class you created
2. **Include examples** - Real usage examples for all major components
3. **Technical depth** - Include implementation details, algorithms, and design decisions
4. **Cross-references** - Link related components together
5. **Keep current** - Update existing documentation when adding new features

## Output Format

Write the DOCUMENTATION.md file directly to the project root. This should be a complete technical reference.

After creating the file, provide a brief summary of:
- Total Python files documented
- Number of functions/classes cataloged
- Documentation sections created

IMPORTANT: This is comprehensive technical documentation. Be thorough and detailed. If a DOCUMENTATION.md file exists, update it to reflect current state, adding new content and updating changed sections.
IMPORTANT: You must write the file to this exact location: {documentation_path}

CRITICAL: You are only allowed to edit DOCUMENTATION.md, but you may read any script or file in the directory.
'''

CLAUDE_MD_PROMPT = """Analyze this Python codebase and create a concise CLAUDE.md file that serves as a quick-reference guide for future ClaudeCode iterations. This should be a streamlined document focused on rapid navigation and understanding. Follow these specific instructions:

## Analysis Requirements

1. **Scan the directory structure** - Identify key Python files and organization
2. **Catalog RELEVANT functions and classes only** - Focus on commonly used and reusable components
3. **Document quick-access imports** - Create copy-paste ready import statements
4. **Identify reusable utilities** - Highlight functions/classes that prevent code duplication
5. **Note project structure** - High-level organization and navigation paths
6. **Track what YOU created** - Focus on ClaudeCode-written code, not external dependencies

## CLAUDE.md Structure to Generate

Create a CLAUDE.md file with these sections:

### Project Overview
- 2-3 sentence description of what this Python project does
- Primary entry points and main modules

### Quick Navigation
```
Key Files:
- src/main.py - Main execution entry point
- src/utils/helpers.py - Common utilities
- src/models/ - Data models and processors
```

### Import Reference Guide
**Quick copy-paste imports for commonly used functions:**
```python
# Utility functions
from src.utils.helpers import helper_function, AnotherHelper

# Data models
from src.models.data_models import DataModel, ProcessingClass

# Main components
from src.main import main_function
```

### Reusable Components
**Functions and classes to reuse (not rewrite):**
- `utility_function()` in `src/utils/helpers.py` - Does X, use when Y
- `DataProcessor` class in `src/models/processor.py` - Handles Z
- Include brief 1-sentence descriptions only

### Common Patterns
- Standard error handling approach used
- Logging pattern (if applicable)
- File I/O conventions
- Any project-specific coding patterns

### Project Structure
```
Brief directory tree showing main organization
```

## Execution Instructions

1. **Be concise** - This is a quick-reference guide, not comprehensive documentation
2. **Focus on frequency** - Include only commonly used components
3. **Copy-paste ready** - All imports should work as-is
4. **Navigation focused** - Help ClaudeCode quickly find what it needs
5. **Reusability first** - Emphasize code that prevents duplication
6. **One-liners** - Descriptions should be brief and actionable

## Output Format

Write the CLAUDE.md file directly to the project root. Keep it under 200 lines if possible.

After creating the file, provide a brief summary of:
- Key reusable components highlighted
- Main navigation paths established

Remember: This is a quick-reference guide optimized for speed and reusability, not comprehensive documentation.

IMPORTANT: Make sure this document is concise and scannable. If a CLAUDE.md file exists, update it intelligently - add new relevant components, remove outdated information, keep it focused and brief.
IMPORTANT: You must write the file to this exact location: {claude_md_path}"""