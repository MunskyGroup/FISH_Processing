import sys
import os
import pickle
# Get the path of the current script (or current working directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Append the parent directory to sys.path
sys.path.append(parent_dir)
import src

def get_integer_list():
    # Prompt the user for input
    user_input = input("Please enter of index of the steps separated by spaces: ")
    
    # Keep asking for input until the user enters a valid input
    while True:
        # Split the input string into a list of strings
        str_list = user_input.split(' ')
        try:
            int_list = [int(num) for num in str_list]
            break
        except ValueError:
            print("Invalid input! Please enter only integers separated by spaces.")
            user_input = input("Please enter of index of the steps separated by spaces: ")
        
    return int_list


def display_prepipeline_steps():
    possible_steps = dir(src.PrePipelineSteps)

    steps = [i for i in possible_steps if not i.startswith("__")]

    for i, step in enumerate(steps):
        print(f'{i}: {step}')

    prepipeline_indexlist = get_integer_list()
    print(prepipeline_indexlist)
    
    prepipelinesteps = [getattr(src.PrePipelineSteps, steps[i])() for i in prepipeline_indexlist]

    return prepipelinesteps


def display_postpipeline_steps():
    possible_steps = dir(src.PostPipelineSteps)

    steps = [i for i in possible_steps if not i.startswith("__")]

    for i, step in enumerate(steps):
        print(f'{i}: {step}')

    postpipeline_indexlist = get_integer_list()

    print(postpipeline_indexlist)

    postpipelinesteps = [getattr(src.PostPipelineSteps, steps[i])() for i in postpipeline_indexlist]

    return postpipelinesteps


def display_pipeline_steps():
    possible_steps = dir(src.PipelineSteps)

    steps = [i for i in possible_steps if not i.startswith("__")]

    for i, step in enumerate(steps):
        print(f'{i}: {step}')

    pipeline_indexlist = get_integer_list()

    pipelinesteps = [getattr(src.PipelineSteps, steps[i])() for i in pipeline_indexlist]

    return pipelinesteps


def display_object_attributes(obj):
    # Get a list of all attributes of the object
    print("Attributes of the object:" + obj.__class__.__name__)
    attributes = vars(obj)
    # Iterate over the attributes and print them
    attr_list = []
    for i, (attr, value) in enumerate(attributes.items()):
        attr_list.append(attr)
        print(f'{i}: {attr}: {value}')

    user_input = input("Please enter the index of the attribute you want to change (c to continue): ")
    while user_input != 'c':
        try:
            attr_index = int(user_input)
            if attr_index >= 0 and attr_index < len(attr_list):
                attr_name = attr_list[attr_index]
                new_value = input(f"Enter the new value for attribute {attr_name}: ")
                setattr(obj, attr_name, new_value)
                print("Updated attributes:")
                for i, (attr, value) in enumerate(attributes.items()):
                    print(f'{i}: {attr}: {value}')
            else:
                print("Invalid index! Please enter a valid index.")
        except ValueError:
            print("Invalid input! Please enter an integer index.")
        user_input = input("Please enter the index of the attribute you want to change (c to continue): ")

    user_input = input("Would you like to add a new attribute? (y/n): ")
    while user_input == 'y':
        new_attr_name = input("Enter the name of the new attribute: ")
        new_attr_value = input("Enter the value of the new attribute: ")
        setattr(obj, new_attr_name, new_attr_value)
        user_input = input("Would you like to add another attribute? (y/n): ")
    
    return obj


def compile_pipeline_package(PipelineStorageLocation: str):
    # Ask for package name
    package_name = input("Enter the name of the pipeline package: ")
    cwd = os.getcwd()

    # Create a directory for the pipeline package
    package_dir = os.path.join(cwd, PipelineStorageLocation, package_name)
    os.makedirs(package_dir, exist_ok=True)

    # Save the instances to pickle files
    PrePipelineSteps = display_prepipeline_steps()
    PostPipelineSteps = display_postpipeline_steps()
    PipelineSteps = display_pipeline_steps()

    # Create instances of the classes
    settings = src.PipelineSettings()
    scope = src.ScopeClass()
    experiment = src.Experiment()
    data = src.PipelineDataClass()

    # Modify the attributes of the instances
    settings = display_object_attributes(settings)
    scope = display_object_attributes(scope)
    experiment = display_object_attributes(experiment)
    data = display_object_attributes(data)

    # Save the instances to pickle files
    settings_file = os.path.join(package_dir, 'settings.pkl')
    scope_file = os.path.join(package_dir, 'scope.pkl')
    experiment_file = os.path.join(package_dir, 'experiment.pkl')
    data_file = os.path.join(package_dir, 'data.pkl')
    prepipeline_steps_file = os.path.join(package_dir, 'prepipeline_steps.pkl')
    postpipeline_steps_file = os.path.join(package_dir, 'postpipeline_steps.pkl')
    pipeline_steps_file = os.path.join(package_dir, 'pipeline_steps.pkl')

    pickle.dump(settings, open(settings_file, 'wb'))
    pickle.dump(scope, open(scope_file, 'wb'))
    pickle.dump(experiment, open(experiment_file, 'wb'))
    pickle.dump(data, open(data_file, 'wb'))
    pickle.dump(PrePipelineSteps, open(prepipeline_steps_file, 'wb'))
    pickle.dump(PostPipelineSteps, open(postpipeline_steps_file, 'wb'))
    pickle.dump(PipelineSteps, open(pipeline_steps_file, 'wb'))


if __name__ == "__main__":
    compile_pipeline_package('Pipelines')
