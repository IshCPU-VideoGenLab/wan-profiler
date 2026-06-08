# Python Rules — wan-profiler

> Claude Code: follow these rules for every Python file you create or edit.

## Language Version
- Target: Python 3.9
- No `match` statements (3.10+)
- No `type X = ...` syntax (3.12+)
- No `ExceptionGroup` (3.11+)
- Walrus operator `:=` allowed only in simple `if` conditions

## Type Hints
- Required on ALL function signatures (parameters + return type)
- Use `typing` imports for 3.9 compatibility: `List`, `Dict`, `Optional`, `Tuple`, `Union`
- Do NOT use `list[str]` syntax (3.9 needs `List[str]`)

## Docstrings
- Google style on all public functions and classes
- Include `Args:`, `Returns:`, and `Raises:` sections where applicable
- One-line docstrings for simple private helpers are fine

## Logging
- Use `logging` module, NEVER `print()` for production code
- `print()` is acceptable ONLY in CLI output meant for the user (e.g., progress bars)
- Logger name: `logging.getLogger(__name__)`

## Structure
- Prefer functions and `dataclasses` over classes with methods
- No inheritance unless genuinely needed
- Keep files under 300 lines. Split if longer.

## Imports
- Standard library first, then third-party, then local
- One import per line for `from` imports
- Absolute imports only (no relative imports)

## Error Handling
- Never bare `except:` — always catch specific exceptions
- Always log exceptions with `logger.exception()` or `logger.error()`

## Memory Safety
- Never load an entire model into memory at once without checking available RAM
- Use `torch.no_grad()` for all inference/profiling operations
- Delete large tensors explicitly and call `gc.collect()` when memory is tight
- Prefer `float16` / `bfloat16` over `float32` for model weights

## Naming
- Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private: prefix with `_`

## Testing
- Test files: `tests/test_<module>.py`
- Test functions: `test_<what_it_tests>`
- Use `pytest` fixtures, not `unittest.TestCase`
