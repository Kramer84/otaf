import otaf

# These tests are to see if the basic functionalities work, by loading the four basic example models. 

def test_load_example_models():
    available_models = {  #previous optimization results:
        "model1_4_dof": otaf.example_models.model1, #tol: 0.31 mult: 1.35 
        "model2_16_dof": otaf.example_models.model2, #tol: 0.16  mult: 1.21
        "model3_30_dof": otaf.example_models.model3, #tol:  mult:  1.26
        "model4_50_dof": otaf.example_models.model4 #tol: 0.21  mult: 1.15
    }
    for model_name, model in available_models.items():
        # Test if the model can be loaded without errors
        assert model is not None, f"Failed to load {model_name}"
        # Test if the model has the expected attributes
        system_of_constraints = model.get_system_of_constraints_assembly_model()
        distribution_function = model.get_distribution_params()
        dimension = int(model.dim)
        assert system_of_constraints is not None, f"{model_name} does not have a system of constraints"
        assert distribution_function is not None, f"{model_name} does not have a distribution function"
        assert dimension > 0, f"{model_name} has an invalid dimension: {dimension}"