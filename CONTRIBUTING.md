# Contributing to incerto

Thank you for your interest in contributing to **incerto**! We welcome contributions from the community.

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/incerto.git
cd incerto
```

### 2. Set Up Development Environment

We use [uv](https://github.com/astral-sh/uv) for package management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install incerto in development mode with dev dependencies
uv pip install -e .
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

## Development Workflow

### Running Tests

We use pytest for testing. Run the full test suite:

```bash
pytest tests/
```

Run specific module tests:

```bash
pytest tests/test_calibration/
pytest tests/test_ood/
```

Run with coverage:

```bash
pytest --cov=incerto --cov-report=term-missing
```

### Code Style

We use `black` for formatting and `ruff` for linting:

```bash
# Format code
black incerto/ tests/

# Lint code
ruff check incerto/ tests/
```

Pre-commit hooks will automatically check style before commits.

### Documentation

Build documentation locally:

```bash
cd docs
sphinx-build -b html source build/html
open build/html/index.html
```

## Contributing Guidelines

### What to Contribute

We welcome:
- **Bug fixes** - Fix issues or improve robustness
- **New methods** - Implement uncertainty quantification methods from research papers
- **Documentation** - Improve guides, examples, or docstrings
- **Tests** - Increase coverage or add edge cases
- **Examples** - Add tutorials or use cases

### Before Submitting

1. **Tests pass**: Ensure `pytest tests/` passes
2. **Code style**: Run `black` and `ruff`
3. **Documentation**: Add docstrings for new functions/classes
4. **Examples**: Add usage examples if introducing new features
5. **Tests**: Add tests for new functionality

### Pull Request Process

1. **Create an issue** (optional but recommended) - Discuss your idea first
2. **Write code** following our style guidelines
3. **Add tests** that cover your changes
4. **Update documentation** if needed
5. **Submit PR** with clear description of changes

### PR Description Template

```markdown
## Description
Brief description of what this PR does

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Other (please describe)

## Testing
Describe how you tested your changes

## Checklist
- [ ] Tests pass locally
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added for new functionality
```

## Code Guidelines

### Docstrings

Use Google-style docstrings:

```python
def my_function(param1: torch.Tensor, param2: float = 1.0) -> torch.Tensor:
    """Short one-line description.

    Longer description with more details.

    Args:
        param1: Description of first parameter
        param2: Description of second parameter (default: 1.0)

    Returns:
        Description of return value

    Example:
        >>> result = my_function(torch.randn(10), param2=2.0)
        >>> result.shape
        torch.Size([10])
    """
    pass
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import Optional, Tuple
import torch

def calibrate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, float]:
    ...
```

### Testing

Write comprehensive tests:

```python
def test_my_feature():
    # Test basic functionality
    result = my_function(input_data)
    assert result.shape == expected_shape

    # Test edge cases
    with pytest.raises(ValueError):
        my_function(invalid_input)

    # Test numerical correctness
    assert torch.allclose(result, expected_result, atol=1e-6)
```

## Implementing New Methods

When adding a new uncertainty quantification method:

1. **Research** - Cite the paper and understand the method
2. **Module** - Place in appropriate module (calibration, ood, conformal, etc.)
3. **API** - Follow existing API patterns (fit/predict, score, etc.)
4. **Tests** - Add comprehensive tests
5. **Documentation** - Add to API reference and write a guide
6. **Example** - Create a usage example

### Example Structure

```python
# incerto/calibration/methods.py

class NewCalibrator(Calibrator):
    """One-line description.

    Longer description with reference to paper:
    Reference: Author et al., "Paper Title", Conference Year
    Link: https://arxiv.org/abs/...

    Args:
        param1: Description
        param2: Description

    Example:
        >>> calibrator = NewCalibrator(param1=value)
        >>> calibrator.fit(val_logits, val_labels)
        >>> calibrated = calibrator.predict(test_logits)
    """

    def __init__(self, param1: float = 1.0):
        self.param1 = param1

    def fit(self, logits: torch.Tensor, labels: torch.Tensor) -> None:
        """Fit calibrator on validation data."""
        ...

    def predict(self, logits: torch.Tensor) -> PredictionDistribution:
        """Apply calibration to logits."""
        ...
```

## Reporting Issues

### Bug Reports

Include:
- Clear description of the bug
- Minimal code to reproduce
- Expected vs actual behavior
- Environment (Python version, PyTorch version, OS)

### Feature Requests

Include:
- Clear description of the feature
- Use case and motivation
- Reference to relevant papers (if applicable)
- Example API you'd like to see

## Community

- **GitHub Issues**: Report bugs and request features
- **GitHub Discussions**: Ask questions and discuss ideas
- **Code of Conduct**: Be respectful and welcoming to all contributors

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue or discussion if you have questions about contributing!

---

Thank you for contributing to incerto! 🎉
