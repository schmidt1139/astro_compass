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
    with open(file_path, 'w') as f:
        for entry in log:
            f.write(entry + '\n')

def write_config_file(params, path_config):
    """Write configuration parameters to a text file for record-keeping."""
    with open(path_config, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

def read_config_file(path_config):
    """Read configuration parameters from a text file."""
    params = {}
    with open(path_config, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ', 1)
            try:
                # Check if value is an array (starts with '[' and ends with ']')
                if value.startswith('[') and value.endswith(']'):
                    # Parse array by removing brackets and splitting by comma
                    array_str = value[1:-1]  # Remove '[' and ']'
                    if array_str.strip():  # Check if not empty
                        # Split by comma and convert each element
                        value = [float(x.strip()) if '.' in x.strip() else int(x.strip()) 
                                for x in array_str.split(',')]
                    else:
                        value = []
                # Try to convert to float or int if possible
                elif '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # Keep as string if conversion fails
            params[key] = value
    return params
