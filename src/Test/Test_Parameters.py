import pytest
import sys
import os

sys.path.append(os.path.join(os.getcwd(), '..'))

from src.Parameters import Parameters, ScopeClass, Experiment, DataContainer, Settings


def setUp():
    Parameters.clear_instances()
    Parameters.initialize_parameters_instances()

def test_singleton_behavior():
    scope1 = ScopeClass()
    scope2 = ScopeClass()
    assert scope1 == scope2

def test_get_all_instances():
    setUp()
    instances = Parameters.get_all_instances()
    print(instances)
    assert len(instances) == 4
    assert(any(isinstance(instance, ScopeClass) for instance in instances))
    assert(any(isinstance(instance, Experiment) for instance in instances))
    assert(any(isinstance(instance, DataContainer) for instance in instances))
    assert(any(isinstance(instance, Settings) for instance in instances))

def test_clear_instances():
    Parameters.clear_instances()
    assert(len(Parameters.get_all_instances()), 0)

def test_validate():
    with pytest.raises(ValueError):
        Parameters.clear_instances()
        Parameters.validate()

    Parameters.initialize_parameters_instances()
    Experiment(initial_data_location='some place over there rainbow')
    Settings(name='I hope this works')
    Parameters.validate()

def test_update_parameters():
    Parameters.update_parameters({'voxel_size_yx': 150, 'name': 'Test'})
    scope = next(instance for instance in Parameters.get_all_instances() if isinstance(instance, ScopeClass))
    settings = next(instance for instance in Parameters.get_all_instances() if isinstance(instance, Settings))
    assert(scope.voxel_size_yx, 150)
    assert(settings.name, 'Test')

# TODO figure out why this doesnt run in coverage
def test_todict():
    scope = ScopeClass()
    scope_dict = scope.todict()
    assert 'voxel_size_yx' in scope_dict.keys()
    assert scope_dict['voxel_size_yx'] == 130

def test_reset():
    scope = ScopeClass(voxel_size_yx=150)
    scope.reset()
    assert scope.voxel_size_yx == 130

def test_get_parameters():
    params = Parameters.get_parameters()
    assert 'voxel_size_yx' in params.keys()
    assert 'name' in params

def test_experiment_validation():
    experiment = Experiment()
    with pytest.raises(ValueError):
        experiment.validate_parameters()

def test_data_container_setattr():
    data_container = DataContainer(total_num_chunks=10)
    settings = next(instance for instance in Parameters.get_all_instances() if isinstance(instance, Settings))
    assert(settings.num_chunks_to_run, 10)



















