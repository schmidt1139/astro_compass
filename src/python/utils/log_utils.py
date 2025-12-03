def log(info, log, flag_report_to_console=False):
    log.append(info)

    if flag_report_to_console:
        print(info)

    return log


def log_parameters(params, test_log, flag_report_to_console=False):
    test_log = log("\n\nParameter Start:", test_log, flag_report_to_console)

    for key, value in params.items():
        line = f"  {key}: {value}"
        test_log = log(line, test_log, flag_report_to_console)

    test_log = log("Parameter End.\n\n", test_log, flag_report_to_console)

    return test_log


def write_log_to_file(file_path, log):
    with open(file_path, "w") as f:
        for entry in log:
            f.write(entry + "\n")


def read_log_from_file(file_path):
    log = []
    with open(file_path, "r") as f:
        for line in f:
            log.append(line)
    return log


def compare_logs(log1, log2):
    """Compare two logs line by line. Return True if they match, False otherwise."""
    if len(log1) != len(log2):
        print(f"Logs differ in length: {len(log1)} vs {len(log2)}")
        return False

    for line1, line2 in zip(log1, log2):
        if line1 != line2:
            print(f"Logs differ:\nLog1: {line1}\nLog2: {line2}")
            return False

    return True


def write_config_file(params, path_config):
    """Write configuration parameters to a text file for record-keeping."""
    with open(path_config, "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")


def read_config_file(path_config):
    """Read configuration parameters from a legacy text file.

    Prefer using :func:`read_toml_config_file` for new code.
    """

    params = {}
    with open(path_config, "r") as f:
        for line in f:
            # check for a comment
            # ignore lines that start with # or empty lines
            if line.strip().startswith("#") or not line.strip():
                continue

            key, value = line.strip().split(": ", 1)
            try:
                # Check if value is a boolean (True/False/true/false)
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                # Check if value is an array (starts with '[' and ends with ']')
                elif value.startswith("[") and value.endswith("]"):
                    # Parse array by removing brackets and splitting by comma
                    array_str = value[1:-1]  # Remove '[' and ']'
                    if array_str.strip():  # Check if not empty
                        # Split by comma and convert each element
                        value = [
                            float(x.strip()) if "." in x.strip() else int(x.strip())
                            for x in array_str.split(",")
                        ]
                    else:
                        value = []
                # Try to convert to float or int if possible
                elif "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string if conversion fails
            params[key] = value

    # Ensure boolean parameters are actually booleans (config might read as strings)
    if isinstance(params.get("randomize_seeds"), str):
        params["randomize_seeds"] = params["randomize_seeds"].lower() in [
            "true",
            "1",
            "yes",
        ]
    if isinstance(params.get("randomize_tofs"), str):
        params["randomize_tofs"] = params["randomize_tofs"].lower() in [
            "true",
            "1",
            "yes",
        ]

    return params


def read_toml_config_file(path_config):
    """Read TOML configuration parameters into a flat dict.

    Uses the standard library ``tomllib`` (Python 3.11+) and returns a
    single-level dictionary of parameters, mirroring the behavior of
    :func:`read_config_file` but with TOML's native typing.
    """

    import os

    import tomli as toml

    with open(path_config, "rb") as f:
        data = toml.load(f)

    params = {}

    def _flatten(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                _flatten(k if prefix is None else f"{prefix}.{k}", v)
        else:
            # For now, use just the final key (no dotted prefix) to keep
            # compatibility with existing flat-parameter usage.
            key = prefix.split(".")[-1] if isinstance(prefix, str) else prefix
            params[key] = obj

    _flatten(None, data)

    # Expand common path-like entries
    for key in (
        "output_dir",
        "path_training_data",
        "path_replay_buffer",
        "path_SAC_model_load",
    ):
        if key in params and isinstance(params[key], str):
            params[key] = os.path.abspath(os.path.expanduser(params[key]))

    return params
