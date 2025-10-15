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
