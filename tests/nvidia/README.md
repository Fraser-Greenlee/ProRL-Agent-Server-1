# SWE-bench Utils Testing

This directory contains comprehensive tests for the SWE-bench utilities, including both unit tests and integration tests that can work with real data from the `__main__` section examples.

## Test Structure

### Test Categories

1. **Unit Tests**: Fast, isolated tests using mocks
2. **Integration Tests**: Tests with real data when available
3. **Real Data Tests**: Tests specifically requiring the actual parquet dataset
4. **Main Examples Tests**: Tests that validate the examples from `utils.py`'s `__main__` section

### Files

- `test_swebench_utils.py` - Original comprehensive unit tests + new real data tests
- `test_swebench_utils_integration.py` - Integration tests using real examples
- `conftest.py` - Pytest fixtures and configuration
- `pytest.ini` - Pytest configuration and markers
- `run_tests.py` - Test runner script with different options

## Running Tests

### Quick Start

```bash
# Run all unit tests (fast, mocked)
python run_tests.py --unit

# Run integration tests (if real data is available)
python run_tests.py --integration

# Run tests with real data specifically
python run_tests.py --real-data

# Test the main examples from utils.py
python run_tests.py --main-examples

# Run all tests
python run_tests.py --all
```

### Using pytest directly

```bash
# Unit tests only
pytest -m "not integration and not slow" tests/nvidia/

# Integration tests
pytest -m integration tests/nvidia/

# Real data tests
pytest -m real_data tests/nvidia/

# Skip slow tests
pytest -m "not slow" tests/nvidia/

# Run with coverage
pytest --cov=openhands.nvidia --cov-report=term-missing tests/nvidia/
```

## Real Data Setup

### Requirements

The real data tests require access to the SWE-bench dataset:
- Path: `/lustre/fsw/portfolios/nvr/users/mingjiel/data/swegym/train.parquet`
- Tests will be automatically skipped if this file is not available

### What Real Data Tests Cover

1. **Data Loading**: Validates the exact data loading pattern from `__main__`
2. **Instance Processing**: Tests numpy array conversion and serialization
3. **Parallel Evaluation**: Tests the parallel async evaluation pattern
4. **Sequential Evaluation**: Tests the sequential evaluation pattern
5. **Docker Image Generation**: Tests with real instance IDs
6. **Configuration**: Tests configuration generation with real data

## Test Examples from __main__

The `__main__` section of `utils.py` contains several patterns that are now testable:

### 1. Data Loading Pattern
```python
dataset = pd.read_parquet("/path/to/train.parquet")
instance = dataset.iloc[0]['instance']
instance = pd.Series(instance)
instance = instance.apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
```

### 2. Parallel Evaluation Pattern
```python
async def run_parallel_async():
    tasks = []
    for idx in range(1):
        inst_clone = instance.copy()
        inst_clone["instance_id"] = f"{instance['instance_id']}_{idx}"
        gold_patch = inst_clone['patch']
        tasks.append(evaluate_agent(gold_patch, inst_clone))
    return await asyncio.gather(*tasks, return_exceptions=True)
```

### 3. Sequential Evaluation Pattern
```python
async def run_sequential_async():
    results = []
    for i in range(2):
        res = await evaluate_agent(mock_patch, instance)
        results.append(res)
    return results
```

## Test Fixtures

The `conftest.py` file provides several useful fixtures:

- `real_dataset`: Loads the actual parquet dataset (skipped if not available)
- `real_instance`: Processed real instance from the dataset
- `mock_patch`: The exact mock patch used in `__main__`
- `minimal_llm_config`: LLM configuration for testing
- `mock_runtime`: Mock runtime for testing
- `sample_evaluation_result`: Sample evaluation result structure

## Test Markers

Tests are organized using pytest markers:

- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.real_data`: Tests requiring real data
- `@pytest.mark.slow`: Slow-running tests
- `@pytest.mark.asyncio`: Async tests

## Mocking Strategy

The tests use a hybrid approach:

1. **Unit Tests**: Mock all external dependencies (runtime, LLM calls, etc.)
2. **Integration Tests**: Use real data structures but mock expensive operations
3. **Real Data Tests**: Use actual data files but mock runtime/network operations

This allows testing the real data flow without requiring expensive Docker containers or LLM API calls.

## Environment Variables

Some tests may require or check for environment variables:

- `OPENAI_API_KEY`: For LLM configuration (can be dummy for tests)
- `EVAL_DOCKER_IMAGE_PREFIX`: Docker image prefix for testing
- `RUN_WITH_BROWSING`: Browsing configuration

## Troubleshooting

### Real Data Not Available
If real data tests are skipped:
1. Check if the parquet file exists at the expected path
2. Run `python run_tests.py --unit` to run tests without real data
3. The existing unit tests provide comprehensive coverage without real data

### Import Errors
Make sure the project root is in your Python path:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/OpenHands_internal"
```

### Async Test Issues
Async tests require proper event loop handling. The test runner and fixtures handle this automatically.

## Contributing

When adding new tests:

1. Use appropriate markers (`@pytest.mark.real_data`, etc.)
2. Add fixtures to `conftest.py` for reusable test data
3. Mock expensive operations (Docker, LLM calls, network)
4. Test both the happy path and error conditions
5. Update this README if adding new test categories

## Examples

### Testing a New Function with Real Data

```python
@pytest.mark.real_data
def test_my_function_with_real_data(real_instance):
    """Test my function with real instance data"""
    result = my_function(real_instance)
    assert result is not None
    assert 'instance_id' in result
```

### Testing Async Patterns

```python
@pytest.mark.asyncio
@pytest.mark.real_data
async def test_async_pattern(real_instance, mock_runtime):
    """Test async pattern from __main__"""
    with patch('module.expensive_function') as mock_func:
        mock_func.return_value = expected_result
        result = await my_async_function(real_instance)
        assert result == expected_result
```
