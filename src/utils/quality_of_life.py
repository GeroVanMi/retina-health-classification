def preflight_check(device: str, experiment_name: str, dev_mode_is_enabled: bool):
    """
    Gives the user some information about the run
    and asks for confirmation, so that they can
    abort the run if they have made a mistake.
    """
    if dev_mode_is_enabled:
        print("RUNNING IN DEVELOPER TESTING MODE. THIS WILL NOT TRAIN THE MODEL.")
        print("To train the model, set DEV_MODE = False in run_experiment.py!")

    print(f"Run: {experiment_name}")
    print(f"Training on {device} device!")

    input("Confirm with Enter or cancel with Ctrl-C:")
