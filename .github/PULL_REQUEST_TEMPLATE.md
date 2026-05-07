# Pull Request

## Summary

Describe the change clearly.

## Type of Change

Select all that apply:

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Test update
- [ ] Refactor
- [ ] Performance improvement
- [ ] Scientific/model equation change
- [ ] Packaging/CI change
- [ ] Other

## Motivation

Explain why this change is needed.

## What Changed

List the main changes:

## Scientific or Model Impact

Does this change affect model equations, parameter decoding, bounds, loss functions, solver behavior, interpolation,
initialization, or output interpretation?

- [ ] No scientific/model behavior change
- [ ] Yes, scientific/model behavior changed

If yes, explain:

```text
Changed assumption/equation:
Reason:
Expected effect:
Compatibility with previous results:
```

## Tests

Describe how this was tested.

* [ ] Existing tests pass
* [ ] New tests added
* [ ] Manual smoke test completed
* [ ] Not tested

Commands run:

```bash
pytest
ruff check .
ruff format --check .
```

## Data and Reproducibility

* [ ] No private, sensitive, clinical, restricted, or unpublished data is included
* [ ] Example data is synthetic or publicly usable
* [ ] Output format changes are documented
* [ ] Config changes are documented

## Documentation

* [ ] README updated
* [ ] Docs updated
* [ ] CHANGELOG updated
* [ ] Not needed

## Checklist

* [ ] Code is focused and does not mix unrelated changes
* [ ] Public functions/classes have useful docstrings where needed
* [ ] Errors and warnings are clear
* [ ] Logging is useful but not noisy
* [ ] Shape-sensitive code has validation or tests
* [ ] New dependencies are justified
* [ ] License compatibility was considered

## Additional Notes

Add anything reviewers should know.
