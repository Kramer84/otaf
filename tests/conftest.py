import pytest

import otaf


@pytest.fixture(scope="session")
def soc_model():
    """Return ``get(name)`` building each example model's SOC model at most once per session."""
    cache = {}

    def get(name):
        if name not in cache:
            model = getattr(otaf.example_models, name)
            cache[name] = model.get_system_of_constraints_assembly_model()
        return cache[name]

    return get
